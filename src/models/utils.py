# Written by Francesco Paissan
from src.models.train_model import train_model
from src.features.utils import load_features

import numpy as np


def load_data(path: str) -> np.ndarray:
    """Loads features from HD"""
    return np.loadtxt(path)

def load_model():
    data_x = load_features("data/processed/unlabeled")
    model = train_model(data_x, "2-norm", n_clusters=61)

    return model