# Written by Francesco Paissan
from src.models.train_model import train_model
from src.features.utils import load_features

import matplotlib.pyplot as plt
import numpy as np
import glob


def proportion_per_cluster(data_x: np.array, y_pred: np.array, type: str, notebook=False):
    """ Plots the proportion for each cluster. """
    material_proportion = []
    for cluster in np.unique(y_pred):
        cluster_samples = data_x[y_pred == cluster]

        material_proportion.append(len(cluster_samples))

    plt.bar(x=[f'C{i + 1}' for i in range(len(np.unique(y_pred)))],
            height=list(map(lambda x: x / sum(material_proportion), material_proportion)))
    plt.title("Proporzione di sample per cluster")
    plt.xticks(rotation='vertical')

    if notebook:
        plt.savefig(f"../reports/figures/proportions_{type}.pdf")
    else:
        plt.savefig(f"reports/figures/proportions_{type}.pdf")


def compress(data, mask):
    """ Masks python list """
    return list((d for d, s in zip(data, mask) if s))


def cluster_vis(interim_path: str, y_pred: np.array, notebook=False):
    """ Visualize clustered spectrograms. """
    file_list = glob.glob(interim_path + '/*')
    raw_data = []
    for i, f in enumerate(file_list):
        interim_data = np.loadtxt(f, delimiter=',', skiprows=1)

        raw_data.append(interim_data)

    for cluster in np.unique(y_pred):
        data_mask = y_pred == cluster
        cluster_interim = compress(raw_data, data_mask)

        for spec in cluster_interim:
            fig = plt.plot(spec[:, 0], spec[:, 1])
            plt.title(f"Spec for cluster C{cluster}")
            plt.xlabel("cm^{-1}")
            plt.ylabel("Raman Intensity")

        if notebook:
            plt.savefig(f"../reports/figures/cluster{cluster + 1}_vis.pdf")
        else:
            plt.savefig(f"reports/figures/cluster{cluster + 1}_vis.pdf")

        plt.close()


def inertia_analysis(feature_filepath: str, notebook=False):
    """
    Trains a set of k-means/k-medoids classifiers to study which is the best cluster count.
    """
    x = load_features(feature_filepath)

    inertia = []
    for nc in range(1, 100):
        model = train_model(x, None, model_type="means")
        inertia.append(model.inertia_)

    plt.plot(range(1, 100), inertia)
    plt.title("Clustering inertia")
    plt.xlabel("number of clusters")
    plt.ylabel("inertia")

    if notebook:
        plt.savefig(f"../reports/figures/inertia_analysis_vis.pdf")
    else:
        plt.savefig(f"reports/figures/inertia_analysis_vis.pdf")
