# Written by Francesco Paissan
from scipy.stats import zscore
from scipy import integrate
import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    """
    Reads data in csv file with values delimited by white spaces and in the raman spec format.
    """
    names = ['wl'] + [f'r{k}c{i}' for k in range(1, 12) for i in range(1, 12)]

    return pd.read_csv(path, delim_whitespace=True, names=names)


def normalize_col(x: pd.Series, col: pd.Series, norm_type: str) -> pd.DataFrame:
    """
    Normalise pd.Series with matter specified by "norm_type".
    norm_type: - integral: normalizes the integral of the signal; - zscore: computes zscore;
    """
    norm_func = {"integral": col / (integrate.trapz(col, x)),
                 "zscore": zscore(col)}

    return norm_func[norm_type]
