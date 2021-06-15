# Written by Francesco Paissan
from src.models.train_model import train_model
from src.features.utils import load_features

import numpy as np
import pickle


def load_data(path: str) -> np.ndarray:
    """Loads features from HD"""
    return np.loadtxt(path)


def load_model(model_type: str):
    if model_type == "medoids":
        data_x = load_features("data/processed/unlabeled")
        model = train_model(data_x, "1-norm", n_clusters=61, model_type="means")
    elif model_type == "means":
        model = pickle.load(open("models/trained_model.pkl", "rb"))

    return model
