from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    return df

def preprocess(df):
    scaler = StandardScaler()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
