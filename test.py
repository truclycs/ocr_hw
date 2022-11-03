import time
import torch
import argparse
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader

from models.definitions.ocr import OCR
from models.definitions.vocab import Vocab
from metrics.metrics import compute_metrics
from utils import load_yaml, translate, abs_path
from dataloader.data_loader import OCRDataset, ClusterRandomSampler, Collator


class Test():
    def __init__(self, config):
        self.config = config

        self.batch_size = 1
        self.result_excel_file = config['save_file']['excel']
        self.result_text_file = config['save_file']['text']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.expected_height = config['dataset']['expected_height']
        self.image_min_width = config['dataset']['image_min_width']
        self.image_max_width = config['dataset']['image_max_width']
        self.data_loader = config['dataloader']

        self.vocab = Vocab(config['vocab'])

        self.model = OCR(len(self.vocab),
                         config['backbone'],
                         config['cnn_args'],
                         config['transformer'],
                         config['seq_modeling']).to(self.device)
        state_dict = torch.load(f=abs_path(config['weights']), map_location=self.device)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval().to(self.device)

        self.test_gen = self.data_gen('dataset/test_{}'.format(config['dataset']['name']),
                                      config['dataset']['data_root'],
                                      config['dataset']['test_annotation'],
                                      masked_language_model=False)

    def predict(self, sample=None):
        predicts = []
        targets = []
        image_files = []
        case = 0
        for batch in self.test_gen:
            batch = self.batch_to_device(batch)
            translated_sentence, prob = translate(batch['image'], self.model, max_seq_length=256)
            predict = self.vocab.batch_decode(translated_sentence.tolist())
            target = self.vocab.batch_decode(batch['tgt_output'].tolist())
            image_files.extend(batch['filenames'])
            predicts.extend(predict)
            targets.extend(target)

            case += 1
            print(case, batch['filenames'])
            print(target)
            print(predict)
            print("==>", target == predict)
            print()

            if sample and len(predicts) > sample:
                break

        return predicts, targets, image_files

    def precision(self, sample=None):
        predicts, targets, image_files = self.predict(sample=sample)
        test.save_predicted_result(targets, predicts, image_files)
        return compute_metrics(predicts, targets, image_files, self.result_text_file)

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
                             expected_height=self.config['dataset']['expected_height'],
                             image_min_width=self.config['dataset']['image_min_width'],
                             image_max_width=self.config['dataset']['image_max_width'])
        sampler = ClusterRandomSampler(dataset, self.batch_size, False)
        collate_fn = Collator(masked_language_model)
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          collate_fn=collate_fn,
                          **self.config['dataloader'])

    def save_predicted_result(self, targets, predicts, images):
        info = defaultdict()
        for image, target, predict in zip(images, targets, predicts):
            image_file = image.split('/')[-1]
            if 'label' not in info:
                info['label'] = {image_file: target}
                info['predict'] = {image_file: predict}
            else:
                info['label'][image_file] = target
                info['predict'][image_file] = predict

        data = pd.ExcelWriter(self.result_excel_file)
        info = pd.DataFrame(info)
        info.to_excel(data)
        data.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vgg_seq2seq.yml')
    args = parser.parse_args()
    config = load_yaml(args.config)
    start_time = time.time()
    test = Test(config)
    cer, wer, aoc, acc = test.precision(len(test.test_gen))
    end_time = time.time()
    total_time = (end_time-start_time) / len(test.test_gen)
    print(f"acc: {acc:.4f} -aoc: {aoc:.4f} -wer: {wer:.4f} -cer: {cer:.4f} -time: {total_time:.4f}")
