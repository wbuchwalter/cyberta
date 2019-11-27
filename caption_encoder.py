import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class CaptionEncoder(nn.Module):
    def __init__(self, n_rkhs, seq_len, hidden_size=1024, device='cpu'):
        super(CaptionEncoder, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for p in self.bert.parameters():
            p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(in_features=768, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=n_rkhs)
        )
        self.bert.to(device)
        self.fc.to(device)
    
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
        
        # sentence representation is the representation of the [CLS] token
        sent_rep = word_level_rep[:, 0]
        sent_rep = sent_rep.to(self.device)
        sent_rep = sent_rep.detach()  # Don't flow gradients through BERT
        sent_rep = self.fc(sent_rep)
        return sent_rep, word_level_rep
