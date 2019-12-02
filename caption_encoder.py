import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class CaptionEncoder(nn.Module):
    def __init__(self, n_rkhs, seq_len, hidden_size=1024, device='cpu'):
        super(CaptionEncoder, self).__init__()
        self.seq_len = seq_len + 2 # account for [CLS] and [SEP] tokens
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for p in self.bert.parameters():
            # Could be interesting to try finetuning a few layers of BERT
            p.requires_grad = False
        self.conv = nn.Conv1d(768, n_rkhs, 1)
        self.conv = self.conv.to(device)
        self.bert = self.bert.to(device)

    
    def forward(self, x):
        batch_size = len(x)
        encodings = [torch.tensor(self.tokenizer.encode(c, add_special_tokens=True)) for c in x]
        padded = torch.stack(
            [torch.cat([e, torch.zeros((self.seq_len - len(e)), dtype=torch.long)]) if len(e) < self.seq_len
            else e[:self.seq_len]
            for e in encodings])
        padded = padded.to(self.device)
        attn_mask = (padded > 0)
        word_level_rep = self.bert(padded, attention_mask=attn_mask)[0]

        word_level_rep = word_level_rep.detach() # Is this needed?
        word_level_rep = self.conv(word_level_rep.permute(0,2,1)).permute(0,2,1)
        
        # sentence representation is the representation of the [CLS] token
        sent_rep = word_level_rep[:, 0]

        
        #sent_rep = sent_rep.to(self.device)
        #sent_rep = sent_rep.detach()  # Don't flow gradients through BERT
        #sent_rep = self.fc(sent_rep)

        # ignore the [CLS] and [SEP] tokens
        word_level_rep = word_level_rep[:, 1:-1]
        return sent_rep, word_level_rep
