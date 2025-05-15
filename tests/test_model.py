import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from train import train_model
import joblib
import os

def test_train_and_save_model():
    train_model()
    assert os.path.exists('model/iris_model.pkl')
    model = joblib.load('model/iris_model.pkl')
    assert hasattr(model, 'predict')
