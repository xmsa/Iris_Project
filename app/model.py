from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

model_classes = {
    "LogisticRegression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "NaiveBayes": GaussianNB,
    "NeuralNetwork": MLPClassifier,
}


def save_model(model, model_name):
    model_path = Path(f"models/{model_name}.pkl")
    joblib.dump(model, model_path)


def load_model(model_name):
    model_path = Path(f"models/{model_name}.pkl")
    model = joblib.load(model_path)
    return model


def load_models():
    models = {model_name: load_model(model_name) for model_name in model_classes.keys()}
    return models


Path(f"models").mkdir(exist_ok=True)
