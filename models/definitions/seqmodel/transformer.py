import torch
import numpy as np
from torch import nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder, num_decoder,
                 dim_feedforward, max_seq, pos_dropout, trans_dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder, num_decoder, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Shape:
            - src: (W, N, C)
            - tgt: (T, N)
            - src_key_padding_mask: (N, S)
            - tgt_key_padding_mask: (N, T)
            - memory_key_padding_mask: (N, S)
            - output: (N, T, E)
        """
        output = self.transformer(self.pos_enc(src * np.sqrt(self.d_model)),
                                  self.pos_enc(self.embed_tgt(tgt) * np.sqrt(self.d_model)),
                                  tgt_mask=self.gen_nopeek_mask(tgt.shape[0]).to(src.device),
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        output = self.fc(output.transpose(0, 1))
        return output

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_encoder(self, src):
        memory = self.transformer.encoder(self.pos_enc(src * np.sqrt(self.d_model)))
        return memory

    def forward_decoder(self, tgt, memory):
        output = self.transformer.decoder(self.pos_enc(self.embed_tgt(tgt) * np.sqrt(self.d_model)),
                                          memory,
                                          tgt_mask=self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device))
        output = self.fc(output.transpose(0, 1))
        return output, memory


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
