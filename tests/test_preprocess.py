import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import load_data, preprocess

def test_load_data():
    df = load_data()
    assert not df.empty

def test_preprocess_shape():
    df = load_data()
    X, y = preprocess(df)
    assert X.shape[0] == len(y)
