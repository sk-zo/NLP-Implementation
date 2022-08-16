import math
import torch
import torch.nn as nn

def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.dropout1(self.actv(self.linear1(self.norm(x))))
        out = self.dropout2(self.linear2(out))
        return out + x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1): 
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.linear_q = nn.Linear(d_model, n_head*self.d_head)
        self.linear_k = nn.Linear(d_model, n_head*self.d_head)
        self.linear_v = nn.Linear(d_model, n_head*self.d_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        n_batch = q.shape[0]
        len_seq = q.shape[1]
        
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        def transform(x):
            return x.view(n_batch, -1, self.n_head, self.d_head).transpose(1,2)
        
        q = transform(q)
        k = transform(k)
        v = transform(v)
        
        q = q / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(2,3))
        if mask != None:
            scores *= mask
        
        attn = self.dropout(self.softmax(scores))
        
        context = torch.matmul(attn, v)
        context = context.view(n_batch, len_seq, -1)
        
        return context
   
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super(EncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, iter, x, mask=None):
        if iter != 0:
            norm_x = self.norm(x)
        else: norm_x = x
        context = self.attn(norm_x, norm_x, norm_x, mask=mask)
        out = self.dropout(context) + x
        return self.feed_forward(out)
    
    
class Ext_Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout=0.1, n_layer=2):
        super(Ext_Encoder, self).__init__()
        self.n_layer = n_layer
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.tr_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, d_ff, n_head, dropout)
                for _ in range(n_layer)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        pos_emb = self.pos_emb.pe[:, :seq_len]
        if mask != None:
            x = x * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.n_layer):
            if mask != None:
                x = self.tr_layers[i](i, x, ~mask)
            else:
                x = self.tr_layers[i](i, x)
            
        x = self.norm(x)
        scores = self.sigmoid(self.linear(x)).squeeze(-1)
        if mask != None:
            scores = scores * mask.float()
        
        return scores