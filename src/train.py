from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess  # ✅ Correct when both files are in src/
import joblib
import os

def train_model():
    df = load_data()
    X, y = preprocess(df)
    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/iris_model.pkl')
    print("Model trained and saved.")

if __name__ == '__main__':
    train_model()
