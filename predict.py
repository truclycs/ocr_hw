import cv2
import time
import torch
import argparse
import numpy as np
from torch import nn
from typing import Dict, Tuple

from utils import abs_path, load_yaml
from models.definitions.ocr import OCR
from models.definitions.vocab import Vocab


class Predictor:
    def __init__(self, config: Dict, image_height: int = 64, image_min_width: int = 32, image_max_width: int = 1024,
                 max_seq: int = 128, sos_token: int = 1, eos_token: int = 2, device: str = 'cpu') -> None:
        super(Predictor, self).__init__()
        self.device = device

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

        self.max_seq = max_seq
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.vocab = Vocab(config['vocab'])

        self.model = OCR(config['vocab_size'],
                         config['backbone'],
                         config['cnn_args'],
                         config['transformer'],
                         config['seq_modeling']).to(config['device'])
        state_dict = torch.load(f=abs_path(config['weights']), map_location=device)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval().to(device)

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor]:
        sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self._resize(image=image)                 # H', W', C
        sample = torch.from_numpy(sample).to(self.device)  # H', W', C
        sample = sample.permute(2, 0, 1).contiguous()      # C', H', W'
        sample = sample.unsqueeze(dim=0)                   # 1, C', H', W'
        sample = sample.float().div(255.)
        return sample,

    def _resize(self, image: np.ndarray) -> np.ndarray:
        width = self.image_height * image.shape[1] // image.shape[0]
        width = (np.ceil(width / 10) * 10).astype(np.int32)
        width = np.clip(width, a_min=self.image_min_width, a_max=self.image_max_width)
        image = cv2.resize(image, dsize=(width, self.image_height))
        return image

    def process(self, sample: torch.Tensor):
        with torch.no_grad():
            feature = self.model.cnn(sample)
            memory = self.model.transformer.forward_encoder(feature)
        return sample, memory

    def postprocess(self, sample, memory):
        translated_sentence = [[self.sos_token]]
        char_probs = [[1]]

        while len(char_probs) <= self.max_seq and not np.any(np.asarray(translated_sentence).T == self.eos_token):
            tgt_inp = torch.LongTensor(translated_sentence).to(self.device)
            output, memory = self.model.transformer.forward_decoder(tgt_inp, memory)
            output = nn.Softmax(dim=-1)(output)
            output = output.detach().cpu().data
            values, indices = torch.topk(output, 4)
            values = list(values[:, -1, 0])
            indices = list(indices[:, -1, 0])
            char_probs.append(values)
            translated_sentence.append(indices)

        translated_sentence = np.asarray(translated_sentence).T
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)

        s = translated_sentence[0].tolist()
        text = self.vocab.decode(s)

        return text, char_probs[0][1:-1]

    def __call__(self, image):
        image, = self.preprocess(image=image)
        image, memory = self.process(image)
        text, char_probs = self.postprocess(image, memory)
        return text, char_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='test/images/test.png')
    parser.add_argument('--config', default='config/vgg_transformer.yml')
    args = parser.parse_args()

    config = load_yaml(args.config)

    detector = Predictor(config=config)
    image = cv2.imread(args.image)

    begin = time.time()
    text, char_probs = detector(image)
    print(f"TIME: {time.time() - begin:.4f}")
    print("TEXT:", text)
    for i, c in enumerate(text):
        print(f'{c}: {char_probs[i]:.4f}')
