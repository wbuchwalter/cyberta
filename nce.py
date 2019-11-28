import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tanh_clip(x, clip_val=10.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


def loss_xent(logits, labels, ignore_index=-1):
    '''
    compute multinomial cross-entropy, for e.g. training a classifier.
    '''
    xent = F.cross_entropy(tanh_clip(logits, 10.), labels,
                           ignore_index=ignore_index)
    lgt_reg = 1e-3 * (logits**2.).mean()
    return xent + lgt_reg


class NCE_MI_MULTI(nn.Module):
    def __init__(self, tclip=20.):
        super(NCE_MI_MULTI, self).__init__()
        self.tclip = tclip

    def _model_scores(self, r_src, r_trg, mask_mat):
        '''
        Compute the NCE scores for predicting r_src->r_trg.

        Input:
          r_src    : (n_batch_gpu, n_rkhs)
          r_trg    : (n_rkhs, n_batch * n_locs)
          mask_mat : (n_batch_gpu, n_batch)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''
        batch_size = mask_mat.size(0)
        n_locs = int(r_trg.size(1) // batch_size)
        n_rkhs = int(r_src.size(1))
        # reshape mask_mat for ease-of-use
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, n_locs).float()
        mask_neg = 1. - mask_pos

        # compute src->trg raw scores for batch on this gpu
        raw_scores = torch.mm(r_src, r_trg).float()
        raw_scores = raw_scores.reshape(batch_size, batch_size, n_locs)
        raw_scores = raw_scores / n_rkhs**0.5
        lgt_reg = 5e-2 * (raw_scores**2.).mean()
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # (n_batch_gpu, n_locs)
        pos_scores = (mask_pos * raw_scores).sum(dim=1)
        # (n_batch_gpu, n_batch, n_locs)
        neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos)
        # (n_batch_gpu, n_batch * n_locs)
        neg_scores = neg_scores.reshape(batch_size, -1)
        # (n_batch_gpu, n_batch * n_locs)
        mask_neg = mask_neg.reshape(batch_size, -1)
        '''
        for each set of positive examples P_i, compute the max over scores
        for the set of negative samples N_i that are shared across P_i
        '''
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]
        '''
        compute a "partial, safe sum exp" over each negative sample set N_i,
        to broadcast across the positive samples in P_i which share N_i
        -- size will be (n_batch_gpu, 1)
        '''
        neg_sumexp = \
            (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        '''
        use broadcasting of neg_sumexp across the scores in P_i, to compute
        the log-sum-exps for the denominators in the NCE log-softmaxes
        -- size will be (n_batch_gpu, n_locs)
        '''
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes
        # compute the final log-softmax scores for NCE...
        nce_scores = pos_shiftexp - all_logsumexp
        return nce_scores, pos_scores, lgt_reg

    def _loss_g2l(self, r_src, r_trg, mask_mat):
        # compute the nce scores for these features
        nce_scores, raw_scores, lgt_reg = \
            self._model_scores(r_src, r_trg, mask_mat)
        loss_g2l = -nce_scores.mean()
        return loss_g2l, lgt_reg

    def forward(self, cap, r1_trg, r7_trg, mask_mat, mode="train"):
        if mode == "train":
            # compute costs for 1->1 prediction
            loss_1t1, lgt_reg_1t1 = self._loss_g2l(cap, r1_trg, mask_mat)
            # Caption 1 -> Img 7
            loss_1t7, lgt_reg_1t7 = self._loss_g2l(cap, r7_trg, mask_mat)
            lgt_reg = (lgt_reg_1t1 + lgt_reg_1t7)
            return loss_1t1, loss_1t7, lgt_reg
        else:
            nce_scores, raw_scores, lgt_reg = \
                self._model_scores(cap, r7_trg, mask_mat)
            return nce_scores, raw_scores



class LossMultiNCE(nn.Module):
    '''
    Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
    '''
    def __init__(self, tclip=10.):
        super(LossMultiNCE, self).__init__()
        # initialize the dataparallel nce computer (magic!)
        self.nce_func = NCE_MI_MULTI(tclip=tclip)
        #self.nce_func = nn.DataParallel(self.nce_func)

    def forward(self, img_r1, img_r7, cap):
        '''
        Compute nce infomax costs for various combos of source/target layers.

        Compute costs in both directions, i.e. from/to both images (x1, x2).

        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''

        # For now, we just do 'uniscale' NCE with just the final output of both 
        # ResNet50 and BERT
        batch_size = img_r1.size(0)
        n_rkhs = img_r1.size(1)

        # (bs, 1, 1, rkhs) -> (bs, rkhs)
        r1_trg = img_r1.reshape(batch_size, n_rkhs).permute(1, 0)
        r7_trg = img_r7.permute(1, 0, 2, 3).reshape(n_rkhs, -1)

        cap = cap.reshape(batch_size, n_rkhs)

        # make masking matrix to help compute nce costs
        mask_mat = torch.eye(batch_size).cuda()

        # Start with r1_x1 -> bert encodings and bert encodings -> r1_x1 ?

        # compute global->local scores and nce costs via nn.Dataparallel
        #n_gpus = torch.cuda.device_count()
        #caption_trg = caption_trg.unsqueeze(dim=0).expand(n_gpus, -1, -1)
       
        loss_1t1, loss_1t7, lgt_reg = self.nce_func(cap, r1_trg, r7_trg, mask_mat)

        return loss_1t1, loss_1t7, lgt_reg
    
    def model_scores(self, cap, img_r7):
        '''
        Compute scores used in the NCE cost (probably for visualization?)
        Input:
          r_glb: (n_batch, n_rkhs)
          r_lcl: (n_batch, n_rkhs, n_locs)
        Output:
          raw_scores: (n_batch, n_locs)
          nce_scores: (n_batch, n_locs) 
        '''
        # make masking matrix to help compute nce costs
        n_batch = cap.size(0)
        n_rkhs = cap.size(1)
        mask_mat = torch.eye(n_batch).cuda()
        # account for use of nn.DataParallel
        n_gpus = torch.cuda.device_count()
        img_r7 = img_r7.permute(1, 0, 2).reshape(n_rkhs, -1)
        #img_r7 = img_r7.unsqueeze(dim=0).expand(n_gpus, -1, -1)
        # compute raw scores and log-softmax NCE scores
        nce_scores, raw_scores = \
            self.nce_func(cap, img_r7, img_r7, mask_mat, mode='viz')
        return nce_scores, raw_scores


def nce_retrieval(encoded_images, encoded_queries, top_k=5):
    batch_size = encoded_images.size(0)
    n_rkhs = encoded_images.size(1)
    n_queries = encoded_queries.size(0)

    # (bs, 1, 1, rkhs) -> (bs, rkhs)
    encoded_images = encoded_images.reshape(batch_size, n_rkhs)
    encoded_images = F.normalize(encoded_images)

    scores = torch.mm(encoded_queries, encoded_images.t())
    cos_sims_idx = torch.sort(scores, dim=1, descending=True)[1]
    cos_sis_idx = torch.flatten(cos_sims_idx[:, :top_k])
    return cos_sis_idx


def nce_retrieval_multiscale(cap, img_r1, img_r7, top_k=5):
    batch_size = encoded_images.size(0)
    n_rkhs = encoded_images.size(1)
    n_queries = encoded_queries.size(0)

    # (bs, 1, 1, rkhs) -> (bs, rkhs)
    encoded_images = encoded_images.reshape(batch_size, n_rkhs)
    encoded_images = F.normalize(encoded_images)

    scores = torch.mm(encoded_queries, encoded_images.t())
    cos_sims_idx = torch.sort(scores, dim=1, descending=True)[1]
    cos_sis_idx = cos_sims_idx[:, :top_k]
    return cos_sis_idx
