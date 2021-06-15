# Written by Francesco Paissan
from src.features.utils import save_feat_files

from progress.bar import ShadyBar
import numpy as np
import glob
import os


def n_ord_interp(x, y, deg=4):
    """
    Returns parameters after deg-degree polynomial interpolation.
    """
    return np.polyfit(x, y, deg)


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

    save_feat_files(np.array(features_set), os.path.join(output_filepath, "peaks_features.csv"))
