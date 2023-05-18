from lib.settings import save_dict
from lib.settings import load_data
from lib.settings import MODEL_INIT
from lib.settings import save_model


def data_preprocess_load():
    data = load_data()
    model = MODEL_INIT()
    save_dict(data, 'data/data.json')
    save_model(model, model_path='data/model.pkl')


if __name__ == '__main__':
    data_preprocess_load()
