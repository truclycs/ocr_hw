from torch import nn

from models.definitions.backbone.backbone import CNN
from models.definitions.seqmodel.seq2seq import Seq2Seq
from models.definitions.seqmodel.transformer import Transformer
from models.definitions.seqmodel.convs2s import ConvSeq2Seq


class OCR(nn.Module):
    def __init__(self, vocab_size, backbone, cnn_args, seq_args, model):
        super(OCR, self).__init__()

        self.cnn = CNN(backbone, **cnn_args)

        self.model = model

        if model == 'transformer':
            self.seq_model = Transformer(vocab_size, **seq_args)
        elif model == 'seq2seq':
            self.seq_model = Seq2Seq(vocab_size, **seq_args)
        elif model == 'convs2s':
            self.seq_model = ConvSeq2Seq(vocab_size, **seq_args)
        else:
            raise("This model is not supported")

    def forward(self, image, tgt_input, tgt_key_padding_mask):
        src = self.cnn(image)
        if self.model == 'transformer':
            return self.seq_model(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        elif self.model == 'seq2seq':
            return self.seq_model(src, tgt_input)
        elif self.model == 'convs2s':
            return self.seq_model(src, tgt_input)
