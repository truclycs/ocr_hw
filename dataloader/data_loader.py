import cv2
import lmdb
import numpy as np
import os
import random
import six
import sys
import torch
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def create_dataset(output_path, root_dir, annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = [line.strip().split('\t') for line in f.readlines()]

    cache, count = {}, 0
    env = lmdb.open(output_path, map_size=1099511627776)
    pbar = tqdm(range(len(annotations)), ncols=100, desc='Create {}'.format(output_path))

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
    def __init__(self, lmdb_path, root_dir, annotation_path, vocab,
                 expected_height=64, image_min_width=32, image_max_width=2048, transform=None):
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

        self.env = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.n_samples = int(self.txn.get('num-samples'.encode()))
        self.build_cluster_indices()

    def build_cluster_indices(self):
        pbar = tqdm(range(self.__len__()),
                    desc='{} build cluster'.format(self.lmdb_path),
                    ncols=100, position=0, leave=True)

        self.cluster_indices = defaultdict(list)
        for i in pbar:
            self.cluster_indices[self.get_bucket(i)].append(i)

    def get_bucket(self, idx):
        key = 'dim-%09d' % idx
        dim_image = self.txn.get(key.encode())
        dim_image = np.fromstring(dim_image, dtype=np.int32)
        height, width = dim_image
        new_width = self._resize(width, height)
        return new_width

    def read_buffer(self, idx):
        image_file = 'image-%09d' % idx
        label_file = 'label-%09d' % idx
        path_file = 'path-%09d' % idx
        label = self.txn.get(label_file.encode()).decode()
        image_path = self.txn.get(path_file.encode()).decode()
        buf = six.BytesIO()
        buf.write(self.txn.get(image_file.encode()))
        buf.seek(0)
        return buf, label, image_path

    def read_data(self, idx):
        buf, label, image_path = self.read_buffer(idx)

        image = Image.open(buf).convert('RGB')
        if self.transform:
            image = self.transform(image)

        image = image.convert('RGB')
        width, height = image.size
        image = image.resize((self._resize(width, height), self.expected_height), Image.ANTIALIAS)
        image = np.asarray(image).transpose(2, 0, 1)
        image = image / 255

        return image, self.vocab.encode(label), image_path

    def _resize(self, width, height):
        new_width = self.expected_height * width // height
        new_width = np.ceil(new_width / 10) * 10
        new_width = np.clip(width, a_min=self.image_min_width, a_max=self.image_max_width)
        return new_width

    def __getitem__(self, idx):
        image, word, image_path = self.read_data(idx)
        return {'image': image, 'word': word, 'image_path': os.path.join(self.root_dir, image_path)}

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
