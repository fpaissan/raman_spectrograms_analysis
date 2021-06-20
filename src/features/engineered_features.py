# Written by Francesco Paissan
from src.features.n_degree_interp import n_ord_interp
from src.features.utils import save_feat_files

from scipy.signal import find_peaks
from progress.bar import ShadyBar
from scipy.stats import entropy
from scipy.stats import zscore
import pandas as pd
import numpy as np
import glob
import os


def stat_features(df, col_name):
    # First-order stats
    var = df[col_name].std()
    mean = df[col_name].mean()
    skew = df[col_name].skew()
    kurt = df[col_name].kurtosis()

    # Second-order stats
    signal_entropy = entropy(np.absolute(df[col_name]))

    # Signal related
    n_peaks = len(find_peaks(df[col_name])[0])
    zero_crossings = len(np.where(np.diff(np.sign(df[col_name])))[0])

    return {
        "var": var,
        "mean": mean,
        "skew": skew,
        "kurt": kurt,
        "entropy": signal_entropy,
        # "n-peaks": n_peaks,
        "zero-cross": zero_crossings,
        "max_height": np.max(df[col_name]),
        "argmax_height": df.wl[np.argmax(df[col_name])],
        # "integral": np.trapz(df[col_name], x=df.wl),
    }


def extract_features(input_filepath: str, output_filepath: str):
    """
    Builds features for clustering.
    """
    file_list = glob.glob(input_filepath + '/*')
    file_list.sort()
    features_set = np.ndarray(shape=(len(file_list), 8))
    with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
        for i, f in enumerate(file_list):
            interim_data = np.loadtxt(f, delimiter=',', skiprows=1)

            y_axis_data = interim_data[:, 1]

            features_set[i, :] = list(
                stat_features(pd.DataFrame({"wl": interim_data[:, 0], "data": y_axis_data}), col_name="data").values())

            bar.next()

    zscore_feats = np.ndarray(shape=features_set.shape)

    for i in range(8):
        zscore_feats[:, i] = zscore(features_set[:, i])

    save_feat_files(zscore_feats, os.path.join(output_filepath, "peaks_features.pkl"))
