# Written by Francesco Paissan
import numpy as np


def load_data(path: str, range_start: int, range_end: int) -> np.ndarray:
    return np.loadtxt(path, skiprows=1, delimiter=',')[range_start:range_end, 1]
