import os
import yaml
import torch
import numpy as np
from PIL import Image
from importlib import import_module
from pathlib import Path
from torch.nn.functional import softmax


def load_yaml(yaml_file):
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config


def abs_path(path):
    return path if Path(path).is_absolute() else str(Path(__file__).parent.joinpath(path))


def eval_config(config):
    def _eval_config(config):
        if isinstance(config, dict):
            if '_base_' in config:
                base_config = _eval_config(config.pop('_base_'))
                base_config = load_yaml(base_config)
                config = {**base_config, **config}

            for key, value in config.items():
                if key not in ['module', 'class']:
                    config[key] = _eval_config(value)

            if 'module' in config and 'class' in config:
                module = config['module']
                class_ = config['class']
                config_kwargs = config.get(class_, {})
                return getattr(import_module(module), class_)(**config_kwargs)

            return config
        elif isinstance(config, list):
            return [_eval_config(ele) for ele in config]
        elif isinstance(config, str):
            return eval(config, {}, original_config)
        else:
            return config

    if isinstance(config, (str, os.PathLike)):
        config = load_yaml(config)

    original_config = config
    config = _eval_config(config)

    if isinstance(config, dict):
        config.pop('modules', None)

    return config


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * w / h)
    new_w = np.ceil(new_w / 10) * 10
    new_w = np.clip(new_w, image_min_width, image_max_width)
    return int(new_w), expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    image = image.convert('RGB')
    w, h = image.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
    image = image.resize((new_w, image_height), Image.ANTIALIAS)
    image = np.asarray(image).transpose(2, 0, 1)
    return image / 255


def process_input(image, image_height, image_min_width, image_max_width):
    image = process_image(image, image_height, image_min_width, image_max_width)
    image = image[np.newaxis, ...]  # add more 1 axis
    image = torch.FloatTensor(image)
    return image


def translate(image, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCxHxW"
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

    return translated_sentence, char_probs
