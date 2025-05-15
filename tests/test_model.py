import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from train import train_model
from preprocess import load_data, preprocess

import joblib
import os

def test_model_accuracy():
    model = train_model()
    df = load_data()
    X, y = preprocess(df)
    score = model.score(X, y)
    assert score > 0.8

def test_train_and_save_model():
    train_model()
    assert os.path.exists('model/iris_model.pkl')
    model = joblib.load('model/iris_model.pkl')
    assert hasattr(model, 'predict')
