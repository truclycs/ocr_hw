import torch
import argparse
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader

from utils import load_yaml
from models.definitions.ocr import OCR
from models.definitions.vocab import Vocab
from metrics.metrics import compute_accuracy, compute_cer, compute_wer
from dataloader.data_loader import OCRDataset, ClusterRandomSampler, Collator


class Test():
    def __init__(self, config):
        self.config = config

        self.batch_size = 1
        self.result_excel_file = config['save_file']['excel']
        self.result_text_file = config['save_file']['text']
        self.load_weights(config['weights'])
        self.backbone = config['backbone']
        self.cnn = config['cnn_args']
        self.transformer = config['transformer']
        self.seq_model = config['seq_modeling']
        self.device = config['device']
        self.vocab = Vocab(config['vocab'])

        self.dataset_name = config['dataset']['name']
        self.data_root = config['dataset']['data_root']
        self.test_annotation = config['dataset']['test_annotation']
        self.expected_height = config['dataset']['expected_height']
        self.image_min_width = config['dataset']['image_min_width']
        self.image_max_width = config['dataset']['image_max_width']
        self.data_loader = config['dataloader']

        self.model = OCR(len(self.vocab), self.backbone, self.cnn, self.transformer, self.seq_model).to(self.device)
        self.test_gen = self.data_gen('dataset/test_{}'.format(self.dataset_name), self.data_root, self.test_annotation, masked_language_model=False)

    def predict(self, sample=None):
        predicts = []
        targets = []
        image_files = []
        case = 0
        for batch in self.test_gen:
            batch = self.batch_to_device(batch)
            translated_sentence, prob = translate(batch['image'], self.model, max_seq_length=256)
            predict_s = self.vocab.batch_decode(translated_sentence.tolist())
            target = self.vocab.batch_decode(batch['tgt_output'].tolist())
            image_files.extend(batch['filenames'])
            predicts.extend(predict_s)
            targets.extend(target)

            case += 1
            print(case)
            print(batch['filenames'])
            print(target)
            print(predict_s, end='\n')

            if sample is not None and len(predicts) > sample:
                break

        return predicts, targets, image_files

    def precision(self, sample=None):
        predicts, targets, image_files = self.predict(sample=sample)
        test.save_predicted_result(targets, predicts, image_files)

        acc_full_seq = compute_accuracy(predicts, targets, mode='full_string', image_files=image_files, file_save=self.result_text_file)
        acc_per_char = compute_accuracy(predicts, targets, mode='per_char', image_files=image_files)

        cer_distances, num_chars = compute_cer(predicts, targets, image_files=image_files)
        wer_distances, num_words = compute_wer(predicts, targets, image_files=image_files)

        cer_distances = torch.sum(cer_distances).float()
        num_chars = torch.sum(num_chars)
        wer_distances = torch.sum(wer_distances).float()
        num_words = torch.sum(num_words)
        cer = cer_distances / num_chars.item()
        wer = wer_distances / num_words.item()

        return acc_full_seq, acc_per_char, cer, wer

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=True)

    def batch_to_device(self, batch):
        image = batch['image'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

        batch = {'image': image,
                 'tgt_input': tgt_input,
                 'tgt_output': tgt_output,
                 'tgt_padding_mask': tgt_padding_mask,
                 'filenames': batch['filenames']}

        return batch

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path,
                             root_dir=data_root, annotation_path=annotation,
                             vocab=self.vocab, transform=transform,
                             expected_height=self.config['dataset']['image_height'],
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
            image = image.split('/')[-1]
            image = image.split('_')
            name = image[-1]
            label = '_'.join(image[i] for i in range(len(image) - 1))

            if name not in info:
                info[name] = {label: target, label + '_res': predict}
            else:
                info[name][label] = target
                info[name][label + '_res'] = predict

        data = pd.ExcelWriter(self.result_excel_file)
        info = pd.DataFrame(info)
        info.to_excel(data)
        data.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vgg-transformer.yml')
    args = parser.parse_args()
    config = load_yaml(args.config)
    test = Test(config)
    acc_full_seq, acc_per_char, cer, wer = test.precision(len(test.test_gen))
    print("acc: {:.4f} - acc per char: {:.4f} - cer {:.4f} - wer {:.4f}".format(acc_full_seq, acc_per_char, cer, wer))
