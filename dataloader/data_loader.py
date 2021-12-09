import os
import sys
import six
import cv2
import lmdb
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utils import process_image, resize


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def create_dataset(output_path, root_dir, annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = [line.strip().split('\t') for line in f.readlines()]

    cache = {}
    count = 0
    env = lmdb.open(output_path, map_size=1099511627776)
    pbar = tqdm(range(len(annotations)), ncols=100, desc=f'Create {output_path}')

    for i in pbar:
        image_file, label = annotations[i]

        with open(os.path.join(root_dir, image_file), 'rb') as f:
            image_bin = f.read()

        image = cv2.imdecode(np.fromstring(image_bin, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        cache['image-%09d' % count] = image_bin
        cache['label-%09d' % count] = label.encode()
        cache['path-%09d' % count] = image_file.encode()
        cache['dim-%09d' % count] = np.array([image.shape[0], image.shape[1]], dtype=np.int32).tobytes()

        count += 1
        if count % 1000 == 0:
            write_cache(env, cache)
            cache = {}

    cache['num-samples'] = str(count - 1).encode()
    write_cache(env, cache)
    print('Created dataset with %d samples' % (count - 1))
    sys.stdout.flush()


class OCRDataset(Dataset):
    def __init__(self, lmdb_path, root_dir, annotation_path, vocab, expected_height: int = 64,
                 image_min_width: int = 64, image_max_width: int = 4096, transform=None):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.expected_height = expected_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width
        self.lmdb_path = lmdb_path

        if os.path.isdir(lmdb_path):
            print('{} exists. Remove folder to create new dataset'.format(lmdb_path))
            sys.stdout.flush()
        else:
            create_dataset(lmdb_path, root_dir, annotation_path)

        self.env = lmdb.open(lmdb_path,
                             max_readers=8,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

        self.txn = self.env.begin(write=False)
        self.n_samples = int(self.txn.get('num-samples'.encode()))
        self.build_cluster_indices()

    def build_cluster_indices(self):
        self.cluster_indices = defaultdict(list)
        pbar = tqdm(range(self.__len__()),
                    desc='{} build cluster'.format(self.lmdb_path),
                    ncols=100,
                    position=0,
                    leave=True)

        for i in pbar:
            bucket = self.get_bucket(i)  # get new_width of image i follow expected_height
            self.cluster_indices[bucket].append(i)  # batch image i to bucket<new_width>

    def get_bucket(self, idx):
        key = 'dim-%09d' % idx
        dim_image = self.txn.get(key.encode())
        dim_image = np.fromstring(dim_image, dtype=np.int32)
        image_height, image_width = dim_image
        new_width, _ = resize(image_width,
                              image_height,
                              self.expected_height,
                              self.image_min_width,
                              self.image_max_width)
        return new_width

    def read_buffer(self, idx):
        image_file = 'image-%09d' % idx
        label_file = 'label-%09d' % idx
        path_file = 'path-%09d' % idx
        image_buf = self.txn.get(image_file.encode())
        label = self.txn.get(label_file.encode()).decode()
        image_path = self.txn.get(path_file.encode()).decode()
        buf = six.BytesIO()
        buf.write(image_buf)
        buf.seek(0)
        return buf, label, image_path

    def read_data(self, idx):
        buf, label, image_path = self.read_buffer(idx)
        image = Image.open(buf).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image_bw = process_image(image, self.expected_height, self.image_min_width, self.image_max_width)
        word = self.vocab.encode(label)
        return image_bw, word, image_path

    def __getitem__(self, idx):
        image, word, image_path = self.read_data(idx)
        image_path = os.path.join(self.root_dir, image_path)
        return {'image': image, 'word': word, 'image_path': image_path}

    def __len__(self):
        return self.n_samples


class ClusterRandomSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_lists = []
        for cluster, cluster_indices in self.data_source.cluster_indices.items():
            batches = [cluster_indices[i: i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            self.batch_lists.extend(batches)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_lists)
            for batch in self.batch_lists:
                random.shuffle(batch)
        return iter(self.batch_lists)

    def __len__(self):
        return len(self.data_source)


class Collator(object):
    def __init__(self, masked_language_model=True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        filenames, image, target_weights, tgt_input = [], [], [], []
        max_label_len = max(len(sample['word']) for sample in batch)

        for sample in batch:
            image.append(sample['image'])
            filenames.append(sample['image_path'])
            label = sample['word']
            label_len = len(label)

            tgt_input.append(np.concatenate((label, np.zeros(max_label_len - label_len, dtype=np.int32))))
            target_weights.append(np.concatenate((np.ones(label_len - 1, dtype=np.float32),
                                                  np.zeros(max_label_len - (label_len - 1), dtype=np.float32))))

        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1] = 0

        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        return {'image': torch.FloatTensor(np.array(image, dtype=np.float32)),
                'tgt_input': torch.LongTensor(tgt_input),
                'tgt_output': torch.LongTensor(tgt_output),
                'tgt_padding_mask': torch.BoolTensor(np.array(target_weights) == 0),
                'filenames': filenames}
