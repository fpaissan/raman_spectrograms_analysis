# Written by Francesco Paissan
import numpy as np


def load_data(path: str) -> np.ndarray:
    return np.loadtxt(path, skiprows=1, delimiter=',')[:, 1]
