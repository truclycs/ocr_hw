from torch import nn
from models.definitions.cnn import CNN
from models.definitions.seqmodel.seq2seq import Seq2Seq
from models.definitions.seqmodel.transformer import Transformer


class OCR(nn.Module):
    def __init__(self, vocab_size, backbone, cnn_args, transformer_args, seq_modeling):
        super(OCR, self).__init__()
        self.cnn = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling
        if seq_modeling == 'transformer':
            self.transformer = Transformer(vocab_size, **transformer_args)
        elif seq_modeling == 'seq2seq':
            self.transformer = Seq2Seq(vocab_size, **transformer_args)

    def forward(self, image, tgt_input, tgt_key_padding_mask):
        src = self.cnn(image)
        if self.seq_modeling == 'transformer':
            return self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        elif self.seq_modeling == 'seq2seq':
            return self.transformer(src, tgt_input)
