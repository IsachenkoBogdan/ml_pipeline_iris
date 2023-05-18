import json
import yaml
import mlflow
import functools
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from importlib import import_module


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def read_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_method_from_lib(full_import: str):
    package, module = full_import.rsplit('.', 1)
    return getattr(import_module(package), module)


def mlflow_logging_with_arguments(func, config, data_path):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        model, metrics, report = func(*args, **kwargs)

        params = config | {'run_type': func.__name__}
        image_save_path = f'data/{params["run_type"]}/heatmap.png'

        print(f'{params["run_type"]} params - {params}')
        print(f'{params["run_type"]} metrics - {metrics}')
        print(report)

        data = load_dict(data_path)
        sns.heatmap(pd.DataFrame(data[f'{params["run_type"]}_x']).corr())
        plt.savefig(image_save_path)

        if params["run_type"] == 'train':
            save_dict(metrics, f'data/{params["run_type"]}/metrics.json')
        else:
            save_dict(metrics, 'data/metrics.json')

        with Image.open(image_save_path) as im:
            mlflow.log_image(im, 'data/heatmap.png')
        mlflow.sklearn.log_model(model, 'data/model.pkl')
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        return model, metrics

    return wrapper_decorator


def parse_metrics_funcs(config_metrics):
    metrics_funcs = {}
    for metric in config_metrics:
        if not isinstance(metric, dict):
            method = get_method_from_lib(metric)
            metrics_funcs |= {method.__name__: method}
        else:
            method = get_method_from_lib(*metric.keys())
            metrics_funcs |= {method.__name__: functools.partial(method, **list(metric.values())[0])}
    return metrics_funcs
