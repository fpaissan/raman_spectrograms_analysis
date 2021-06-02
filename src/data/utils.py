# Written by Francesco Paissan
from scipy import integrate
import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    """
    Reads data in csv file with values delimited by white spaces and in the raman spec format.
    """
    names = ['wl'] + [f'r{k}c{i}' for k in range(1, 12) for i in range(1, 12)]

    return pd.read_csv(path, delim_whitespace=True, names=names)


def normalize_col(x, col) -> pd.DataFrame:
    return col / (integrate.trapz(col, x))
