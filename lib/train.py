from lib.settings import mlflow_logging
from lib.settings import METRICS_DICT
from lib.settings import load_dict
from lib.settings import load_saved_model
from lib.settings import save_model
from sklearn.metrics import classification_report


@mlflow_logging
def train():
    model = load_saved_model(model_path='data/model.pkl')
    data = load_dict('data/data.json')

    x = [*data['train_x'], *data['eval_x']]
    y = [*data['train_y'], *data['eval_y']]
    model.fit(data['train_x'], data['train_y'])

    preds = model.predict(x)
    metrics = {name: metric(y, preds) for name, metric in METRICS_DICT.items()}

    save_model(model)

    return model, metrics, classification_report(y, preds)


if __name__ == '__main__':
    train()
