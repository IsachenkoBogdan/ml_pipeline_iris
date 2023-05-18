import os
import pickle
import random
from functools import partial
import mlflow
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from lib.helper_funcs import get_method_from_lib
from lib.helper_funcs import load_dict
from lib.helper_funcs import save_dict
from lib.helper_funcs import mlflow_logging_with_arguments
from lib.helper_funcs import read_config
from lib.helper_funcs import parse_metrics_funcs

METRICS_PKG = 'sklearn.metrics'
MODEL_PATH = 'data/train/model.pkl'
DATA_PATH = 'data/data.json'

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('isachenko_bv_experiments')

RANDOM_SEED = 1
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONFIG = read_config('params.yaml')
MODEL_INIT = partial(get_method_from_lib(CONFIG['train']['model']['model_class']),
                     **CONFIG['train']['model']['params'])

METRICS_DICT = parse_metrics_funcs(CONFIG['metrics'])

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/eval', exist_ok=True)

mlflow_logging = partial(mlflow_logging_with_arguments, config=CONFIG, data_path=DATA_PATH)


def load_saved_model(model_path=MODEL_PATH):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)


def load_data(features=CONFIG['train']['features'], test_size=CONFIG['train']['test_size']):
    x, y = datasets.load_iris(return_X_y=True, as_frame=True)
    x = x[features].values.tolist()
    y = y.values.tolist()
    train_x, eval_x, train_y, eval_y = train_test_split(x, y, test_size=test_size)
    data = {
        'train_x': train_x,
        'eval_x': eval_x,
        'train_y': train_y,
        'eval_y': eval_y,
    }
    return data


def save_model(model, model_path=MODEL_PATH):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
