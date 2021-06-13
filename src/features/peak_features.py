# Written by Francesco Paissan
from src.features.utils import save_feat_files

from scipy.signal import find_peaks
from progress.bar import ShadyBar
import numpy as np
import glob
import os


def find_maxpeak_argmax(input_filepath, output_filepath):
    file_list = glob.glob(input_filepath + '/*')
    file_list.sort()
    features_set = []
    with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
        for f in file_list:
            interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
            features_set.append(np.argmax(interim_data[:, 1]))

            bar.next()

    np.savetxt(os.path.join(output_filepath, "peaks_features.csv"), np.array(features_set))


def find_maxpeak_2d(input_filepath, output_filepath):
    file_list = glob.glob(input_filepath + '/*')
    file_list.sort()
    features_set = []
    with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
        for f in file_list:
            interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
            features_set.append([np.argmax(interim_data[:, 1]), interim_data[np.argmax(interim_data[:, 1]), 1]])

            bar.next()

    np.savetxt(os.path.join(output_filepath, "peaks_features.csv"), np.array(features_set))


def find_maxpeak_2d(input_filepath, output_filepath):
    file_list = glob.glob(input_filepath + '/*')
    file_list.sort()
    features_set = []
    with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
        for f in file_list:
            interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
            features_set.append([np.argmax(interim_data[:, 1]), interim_data[np.argmax(interim_data[:, 1]), 1]])

            bar.next()

    np.savetxt(os.path.join(output_filepath, "peaks_features.csv"), np.array(features_set))


def gen_n_peak(n_peaks, type):
    def find_n_maxpeak_argmax(input_filepath, output_filepath):
        file_list = glob.glob(input_filepath + '/*')
        file_list.sort()
        features_set = []
        with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
            for f in file_list:
                interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
                y_axis_data = interim_data[:, 1]

                indexes = find_peaks(y_axis_data, distance=30)[0]
                y_axis_peaks = y_axis_data[indexes]

                n_peaks_x = y_axis_peaks.argsort()[::-1][:n_peaks]
                interim_data_indexes = indexes[n_peaks_x]

                features_set.append(interim_data[interim_data_indexes, 0])

                bar.next()

        np.savetxt(os.path.join(output_filepath, "peaks_features.csv"), np.array(features_set))

    def find_n_maxpeak(input_filepath, output_filepath):
        file_list = glob.glob(input_filepath + '/*')
        file_list.sort()
        features_set = np.ndarray(shape=(len(file_list), 2, n_peaks))
        with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
            for i, f in enumerate(file_list):
                interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
                y_axis_data = interim_data[:, 1]

                indexes = find_peaks(y_axis_data, distance=30)[0]
                y_axis_peaks = y_axis_data[indexes]

                n_peaks_x = y_axis_peaks.argsort()[::-1][:n_peaks]
                interim_data_indexes = indexes[n_peaks_x]

                features_set[i, 0, :] = interim_data[interim_data_indexes, 0]
                features_set[i, 1, :] = interim_data[interim_data_indexes, 1]

                bar.next()

        save_feat_files(features_set, os.path.join(output_filepath, "peaks_features.csv"))

    if type == "argmax":
        return find_n_maxpeak_argmax
    elif type == "amp":
        return find_n_maxpeak
