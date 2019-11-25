import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class CaptionEncoder(nn.Module):
    def __init__(self, n_rkhs, seq_len, device):
        super(CaptionEncoder, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.tokenizer = tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Sequential(
            nn.Linear(in_features=seq_len * 768, out_features=seq_len * 768 // 2),
            nn.ReLU(),
            nn.Linear(in_features=seq_len * 768 // 2, out_features=n_rkhs)
        )
        self.fc.to(device)
    
    def forward(self, x):
        batch_size = len(x)
        encodings = [torch.tensor(self.tokenizer.encode(c)) for c in x]
        padded = torch.stack(
            [torch.cat([e, torch.zeros((self.seq_len - len(e)), dtype=torch.long)]) if len(e) < self.seq_len
            else e[:self.seq_len]
            for e in encodings])
        attn_mask = (padded > 0)
        out = self.bert(padded, attention_mask=attn_mask)[0]
        out = out.reshape(batch_size, 768 * self.seq_len)
        out = out.to(self.device)
        out = out.detach() # Don't flow gradients through BERT
        out = self.fc(out)
        return out