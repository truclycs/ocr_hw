import argparse
import imgaug as ia
import numpy as np
import os
import torch
from datetime import datetime
from imgaug import augmenters as iaa
from PIL import Image
from torch.nn.functional import softmax
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dataloader.data_loader import ClusterRandomSampler, Collator, OCRDataset
from loss.labelsmoothingloss import LabelSmoothingLoss
from metrics.metrics import compute_accuracy, compute_cer, compute_wer
from models.definitions.ocr import OCR
from models.definitions.vocab import Vocab
from utils import load_yaml


class Logger():
    def __init__(self, fname):
        path, _ = os.path.split(fname)
        os.makedirs(path, exist_ok=True)
        self.logger = open(fname, 'w')

    def log(self, string):
        self.logger.write(string + '\n')
        self.logger.flush()

    def close(self):
        self.logger.close()


class ImageAugTransform:
    def __init__(self):
        sometimes = self.sometimes()
        self.aug = iaa.Sequential(iaa.SomeOf((1, 5), [
            # blur
            sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)), iaa.MotionBlur(k=3)])),
            # color
            sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
            sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
            sometimes(iaa.Invert(0.25, per_channel=0.5)),
            sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
            sometimes(iaa.Dropout2d(p=0.5)),
            sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
            sometimes(iaa.Add((-40, 40), per_channel=0.5)),
            sometimes(iaa.JpegCompression(compression=(5, 80))),
            # distort
            sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
            sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), order=[0, 1], cval=(0, 255), mode=ia.ALL)),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
            sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)), iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])), ],
            random_order=True), random_order=True)

    def sometimes(self):
        return lambda aug: iaa.Sometimes(0.3, aug)

    def __call__(self, image):
        image = np.array(image)
        image = self.aug.augment_image(image)
        image = Image.fromarray(image)
        return image


class Trainer():
    def __init__(self, config, pretrained=True, augmentor=ImageAugTransform()):
        self.backbone = config['backbone']
        self.cnn = config['cnn_args']
        self.transformer = config['transformer']
        self.seq_model = config['seq_modeling']
        self.device = config['device']
        self.vocab = Vocab(config['vocab'])

        self.dataset_name = config['dataset']['name']
        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.expected_height = config['dataset']['expected_height']
        self.image_min_width = config['dataset']['image_min_width']
        self.image_max_width = config['dataset']['image_max_width']
        self.data_loader = config['dataloader']

        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']

        self.num_iters = config['trainer']['iters']
        self.batch_size = config['trainer']['batch_size']
        self.display_cycle = config['trainer']['display_cycle']
        self.valid_cycle = config['trainer']['valid_cycle']

        cur_time = datetime.now().strftime('%y%m%d%H%M')
        self.weight_dir = config['trainer']['weight_dir'] + cur_time + '/'
        self.save_checkpoint = self.weight_dir + 'checkpoint.pth'
        self.logger = Logger(self.weight_dir + 'logger.log')

        self.model = OCR(len(self.vocab), self.backbone, self.cnn, self.transformer, self.seq_model).to(self.device)

        if pretrained:
            weight_file = config['weights']
            self.load_weight(weight_file)

        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])
        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        transforms = None
        if self.image_aug:
            transforms = augmentor

        self.train_gen = self.data_gen(f'dataset/train_{self.dataset_name}', self.data_root,
                                       self.train_annotation, self.masked_language_model, transform=transforms)

        self.valid_gen = self.data_gen(f'dataset/valid_{self.dataset_name}', self.data_root,
                                       self.valid_annotation, masked_language_model=False)

    def train(self):
        total_loss = best_acc = 0
        data_iter = iter(self.train_gen)

        for i in range(1, self.num_iters + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            loss = self.step(batch)
            total_loss += loss

            if i % self.display_cycle == 0:
                cur_loss = total_loss / self.display_cycle
                info = 'iter: {:06d} - loss: {:.4f} - lr: {:.5f}'.format(i, cur_loss, self.optimizer.param_groups[0]['lr'])
                print(info)
                self.logger.log(info)
                total_loss = 0

            if self.valid_annotation and i % self.valid_cycle == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char, cer, wer = self.precision(len(self.valid_annotation))

                info = 'i: {:06d} - valid loss: {:.3f} - acc: {:.4f} - apc: {:.3f} - cer {:.3f} - wer {:.3f}'.format(i, val_loss, acc_full_seq, acc_per_char, cer, wer)
                print(info)
                self.logger.log(info)

                if acc_full_seq > best_acc:
                    self.save_weights(self.weight_dir + datetime.now().strftime('%y%m%d%H%M') + ".pth")
                    best_acc = acc_full_seq

                self.visualize_prediction()

            self.save_weights(self.save_checkpoint)

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)

        outputs = self.model(batch['image'], batch['tgt_input'], tgt_key_padding_mask=batch['tgt_padding_mask'])
        outputs = outputs.view(-1, outputs.size(2))
        tgt_output = batch['tgt_output'].view(-1)

        loss = self.criterion(outputs, tgt_output)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validate(self):
        self.model.eval()

        total_loss = []
        with torch.no_grad():
            for batch in self.valid_gen:
                batch = self.batch_to_device(batch)

                outputs = self.model(batch['image'], batch['tgt_input'], batch['tgt_padding_mask'])
                outputs = outputs.flatten(0, 1)

                tgt_output = batch['tgt_output'].flatten()

                loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())

        self.model.train()

        return np.mean(total_loss)

    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        image_files = []
        probs = []
        for batch in self.valid_gen:
            batch = self.batch_to_device(batch)
            translated_sentence, prob = self.translate(batch['image'], self.model, max_seq_length=256)
            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())
            image_files.extend(batch['filenames'])
            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            probs.extend(prob)
            if sample is not None and len(pred_sents) > sample:
                break
        return pred_sents, actual_sents, image_files, probs

    def translate(self, image, model, max_seq_length=128, sos_token=1, eos_token=2):
        model.eval()

        with torch.no_grad():
            memory = model.transformer.forward_encoder(model.cnn(image))
            translated_sentence = [[sos_token] * len(image)]
            char_probs = [[1] * len(image)]

            max_length = 0
            while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
                tgt_inp = torch.LongTensor(translated_sentence).to(image.device)
                output, memory = model.transformer.forward_decoder(tgt_inp, memory)
                output = softmax(output, dim=-1)
                output = output.to('cpu')
                values, indices = torch.topk(output, 5)
                indices = list(indices[:, -1, 0])
                values = list(values[:, -1, 0])
                char_probs.append(values)
                translated_sentence.append(indices)
                max_length += 1

            translated_sentence = np.asarray(translated_sentence).T
            char_probs = np.asarray(char_probs).T
            char_probs = np.multiply(char_probs, translated_sentence > 3)
            char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

        return translated_sentence, char_probs

    def precision(self, sample=None):
        predicts, targets, _, _ = self.predict(sample=sample)

        acc_full_seq = compute_accuracy(predicts, targets, mode='full_string')
        acc_per_char = compute_accuracy(predicts, targets, mode='per_char')

        cer_distances, num_chars = compute_cer(predicts, targets)
        wer_distances, num_words = compute_wer(predicts, targets)

        cer_distances = torch.sum(cer_distances).float()
        num_chars = torch.sum(num_chars)

        wer_distances = torch.sum(wer_distances).float()
        num_words = torch.sum(num_words)

        cer = cer_distances / num_chars.item()
        wer = wer_distances / num_words.item()

        return acc_full_seq, acc_per_char, cer, wer

    def visualize_prediction(self, sample=16, errorcase=False):
        pred_sents, actual_sents, image_files, probs = self.predict(sample)
        if errorcase:
            wrongs = []
            for i in range(len(image_files)):
                if pred_sents[i] != actual_sents[i]:
                    wrongs.append(i)

            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            image_files = [image_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]

        for vis_idx in range(0, min(len(image_files), sample)):
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            print("Actuals: ", actual_sent)
            print("Predict: ", pred_sent)
            print()

    def load_weight(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))
        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]
        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path = os.path.split(filename)[0]
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):
        return {'image': batch['image'].to(self.device, non_blocking=True),
                'tgt_input': batch['tgt_input'].to(self.device, non_blocking=True),
                'tgt_output': batch['tgt_output'].to(self.device, non_blocking=True),
                'tgt_padding_mask': batch['tgt_padding_mask'].to(self.device, non_blocking=True),
                'filenames': batch['filenames']}

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path,
                             root_dir=data_root, annotation_path=annotation,
                             vocab=self.vocab, transform=transform,
                             expected_height=self.expected_height,
                             image_min_width=self.image_min_width,
                             image_max_width=self.image_max_width)
        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, **self.data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vgg_transformer.yml')
    args = parser.parse_args()
    config = load_yaml(args.config)

    trainer = Trainer(config=config, pretrained=True)
    trainer.train()
