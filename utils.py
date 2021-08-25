import os
import yaml
from importlib import import_module
from pathlib import Path


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
