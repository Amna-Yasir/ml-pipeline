from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess
import joblib
import os

def train_model():
    df = load_data()
    X, y = preprocess(df)

    clf = RandomForestClassifier()
    clf.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/iris_model.pkl")

    print("Model trained and saved.")
    return clf

if __name__ == "__main__":
    train_model()
