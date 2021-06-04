# Written by Francesco Paissan
import numpy as np


def load_data(path: str) -> np.ndarray:
    """Loads features from HD"""
    return np.loadtxt(path).reshape(-1, 1)
