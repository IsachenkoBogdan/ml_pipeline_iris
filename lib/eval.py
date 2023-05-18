from lib.settings import mlflow_logging
from lib.settings import METRICS_DICT
from lib.settings import load_saved_model
from lib.settings import load_dict
from sklearn.metrics import classification_report


@mlflow_logging
def eval():
    model = load_saved_model()
    data = load_dict('data/data.json')
    preds = model.predict(data['eval_x'])
    metrics = {name: metric(data['eval_y'], preds) for name, metric in METRICS_DICT.items()}
    return model, metrics, classification_report(data['eval_y'], preds)


if __name__ == '__main__':
    eval()
