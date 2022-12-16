import os
import torch
import shutil
import argparse
import numpy as np
from datetime import datetime
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from utils import load_yaml, translate
from models.definitions.ocr import OCR
from models.definitions.vocab import Vocab
from metrics.metrics import compute_metrics
from loss.labelsmoothingloss import LabelSmoothingLoss
from dataloader.data_augmentation import ImageAugTransform
from dataloader.data_loader import ClusterRandomSampler, Collator, OCRDataset


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


class Trainer():
    def __init__(self, config_path, pretrained=True, augmentor=ImageAugTransform()):
        config = load_yaml(config_path)
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
        self.save_checkpoint = self.weight_dir + 'cur_checkpoint.pth'
        self.logger = Logger(self.weight_dir + 'logger.log')

        shutil.copy(config_path, self.weight_dir)

        self.model = OCR(len(self.vocab),
                         config['backbone'],
                         config['cnn_args'],
                         config['transformer'],
                         config['seq_modeling']).to(self.device)

        if pretrained:
            weight_file = config['weights']
            self.load_weight(weight_file)

        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])
        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        transforms = None
        if self.image_aug:
            transforms = augmentor

        self.train_gen = self.data_gen(f'dataset/train_{self.dataset_name}',
                                       self.data_root,
                                       self.train_annotation,
                                       self.masked_language_model,
                                       transform=transforms)

        self.valid_gen = self.data_gen(f'dataset/valid_{self.dataset_name}',
                                       self.data_root,
                                       self.valid_annotation,
                                       masked_language_model=False)

    def train(self):
        total_loss = 0
        best_loss = 10 ** 5
        best_acc = 0
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
                info = '{:06d} - loss: {:.4f} - lr: {:4f}'.format(i, cur_loss, self.optimizer.param_groups[0]['lr'])
                print(info)
                self.logger.log(info)
                total_loss = 0

            if self.valid_gen and i % self.valid_cycle == 0:
                val_loss = self.validate()
                cer, wer, aoc, acc = self.precision(len(self.valid_gen))

                info = f'{i:06d} -loss {val_loss:.4f} -acc {acc:.4f} -aoc {aoc:.4f} -wer {wer:.4f} -cer {cer:.4f}'
                print(info)
                self.logger.log(info)

                if acc > best_acc:
                    # self.save_weights(self.weight_dir + datetime.now().strftime('%y%m%d%H%M') + "_acc.pth")
                    self.save_weights(self.weight_dir + "best_acc.pth")
                    best_acc = acc

                if val_loss < best_loss:
                    # self.save_weights(self.weight_dir + datetime.now().strftime('%y%m%d%H%M') + "_loss.pth")
                    self.save_weights(self.weight_dir + "best_loss.pth")
                    best_loss = val_loss

                self.visualize_prediction()
            self.save_weights(self.save_checkpoint)

    def step(self, batch):
        self.model.train()
        batch = self.batch_to_device(batch)
        outputs = self.model(batch['image'],
                             batch['tgt_input'],
                             tgt_key_padding_mask=batch['tgt_padding_mask'])
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
                outputs = self.model(batch['image'],
                                     batch['tgt_input'],
                                     tgt_key_padding_mask=batch['tgt_padding_mask'])
                outputs = outputs.flatten(0, 1)
                tgt_output = batch['tgt_output'].flatten()
                loss = self.criterion(outputs, tgt_output)
                total_loss.append(loss.item())

                del outputs
                del loss

        total_loss = np.mean(total_loss)

        self.model.train()

        return total_loss

    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        image_files = []
        probs = []
        for batch in self.valid_gen:
            batch = self.batch_to_device(batch)
            translated_sentence, prob = translate(batch['image'], self.model, max_seq_length=256)
            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())
            image_files.extend(batch['filenames'])
            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            probs.extend(prob)
            if sample and len(pred_sents) > sample:
                break
        return pred_sents, actual_sents, image_files, probs

    def precision(self, sample=None):
        predicts, targets, image_files, probs = self.predict(sample=sample)
        return compute_metrics(predicts, targets, image_files)

    def visualize_prediction(self, sample=10):
        pred_sents, actual_sents, image_files, probs = self.predict(sample)
        for vis_idx in range(0, min(len(image_files), sample)):
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            print("Actuals: ", actual_sent)
            print("Predict: ", pred_sent)
            print("==>", actual_sent == pred_sent)

    def load_weight(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))
        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatch shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
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
                             root_dir=data_root,
                             annotation_path=annotation,
                             vocab=self.vocab,
                             transform=transform,
                             expected_height=self.expected_height,
                             image_min_width=self.image_min_width,
                             image_max_width=self.image_max_width)
        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)
        # return DataLoader(dataset,
        #                   batch_sampler=sampler,
        #                   collate_fn=collate_fn,
        #                   **self.data_loader)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=sampler,
                          collate_fn=collate_fn,
                          shuffle=False,
                          drop_last=False,
                          **self.data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vgg_seq2seq.yml')
    args = parser.parse_args()
    trainer = Trainer(config_path=args.config, pretrained=True)
    trainer.train()
