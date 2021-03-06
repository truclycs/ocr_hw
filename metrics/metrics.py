import torch
import editdistance as ed
from typing import List


def compute_cer(predicts: List[str], targets: List[str]) -> float:
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    distances = torch.tensor([ed.distance(predict, target) for predict, target in zip(predicts, targets)])
    num_references = torch.tensor(list(map(len, targets)))

    cer = torch.sum(distances).float() / torch.sum(num_references).item()

    return cer


def compute_wer(predicts: List[str], targets: List[str]) -> float:
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    distances = []
    num_references = []
    for predict, target in zip(predicts, targets):
        predict = predict.split(' ')
        target = target.split(' ')
        distances.append(ed.distance(predict, target))
        num_references.append(len(target))

    distances = torch.tensor(distances)
    num_references = torch.tensor(num_references)

    wer = torch.sum(distances).float() / torch.sum(num_references).item()

    return wer


def compute_accuracy(predicts: List[str], targets: List[str], image_files=None, file_save=None) -> float:
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    correct_count = 0
    wrong_case = []
    for predict, target, image_file in zip(predicts, targets, image_files):
        if predict == target:
            correct_count += 1
        else:
            wrong_case.append(str(image_file) + '\t' + str(target) + '\t' + str(predict) + '\n')
    acc = correct_count / len(targets)

    if file_save:
        with open(file_save, "a") as fw:
            for x in wrong_case:
                fw.write(x)
            fw.write(str(acc))

    return acc


def compute_metrics(predicts: List[str], targets: List[str], image_files=None, file_save=None):
    cer = compute_cer(predicts, targets)
    wer = compute_wer(predicts, targets)
    aoc = 1 - cer
    acc = compute_accuracy(predicts, targets, image_files, file_save)
    return cer, wer, aoc, acc
