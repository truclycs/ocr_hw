import editdistance as ed
import numpy as np
import torch
from typing import List


def compute_cer(predicts: List[List[str]], targets: List[List[str]]):
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    distances = torch.tensor([ed.distance(predict, target) for predict, target in zip(predicts, targets)])
    num_references = torch.tensor(list(map(len, targets)))

    return distances, num_references


def compute_wer(predicts: List[List[str]], targets: List[List[str]]):
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    distances = []
    num_references = []
    for predict, target in zip(predicts, targets):
        predict = ''.join(predict).split(' ')
        target = ''.join(target).split(' ')
        distances.append(ed.distance(predict, target))
        num_references.append(len(target))

    distances = torch.tensor(distances)
    num_references = torch.tensor(num_references)

    return distances, num_references


def compute_accuracy(predicts: List[List[str]], targets:  List[List[str]], mode: str = 'full_string'):
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    if mode == 'per_char':
        accuracy = []
        for index, label in enumerate(targets):
            prediction = predicts[index]
            correct_count = 0
            for i in range(min(len(label), len(prediction))):
                if label[i] == prediction[i]:
                    correct_count += 1
            accuracy.append(correct_count / len(label))
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

    elif mode == 'full_string':
        correct_count = 0
        for index, label in enumerate(targets):
            prediction = predicts[index]
            if prediction == label:
                correct_count += 1
        avg_accuracy = correct_count / len(targets)

    return avg_accuracy
