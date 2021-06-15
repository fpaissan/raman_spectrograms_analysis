# Written by Francesco Paissan
from src.features.utils import save_feat_files

from scipy.interpolate import interp1d
from progress.bar import ShadyBar
import numpy as np
import glob
import os


def n_ord_interp(x, y, deg=5):
    """
    Returns parameters after deg-degree polynomial interpolation.
    """
    return np.polyfit(x, y, deg)


def linear_int(x, y, mode="interp1d"):
    """
    Returns clean spectrogram using linear interpolation
    """
    if mode == "interp1d":
        fit = interp1d(x, y, fill_value="extrapolate")
    else:
        params = n_ord_interp(x, y)
        fit = np.poly1d(params)

    x = np.arange(0, 2400)

    return fit(x)


def fit_params(input_filepath, output_filepath):
    """
    Extract features and saves them in output_filepath folder.
    """
    file_list = glob.glob(input_filepath + '/*')
    file_list.sort()
    features_set = []
    with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
        for f in file_list:
            interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
            features_set.append(n_ord_interp(interim_data[:, 0], interim_data[:, 1]))

            bar.next()

    save_feat_files(np.array(features_set), os.path.join(output_filepath, "peaks_features.pkl"))


def clean_spec(input_filepath, output_filepath):
    """
    Extract features and saves them in output_filepath folder.
    """
    file_list = glob.glob(input_filepath + '/*')
    file_list.sort()
    features_set = []
    with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
        for f in file_list:
            interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
            features_set.append(linear_int(interim_data[:, 0], interim_data[:, 1]))

            bar.next()

    save_feat_files(np.array(features_set), os.path.join(output_filepath, "peaks_features.pkl"))
