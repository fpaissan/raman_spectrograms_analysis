# Written by Francesco Paissan
from scipy.spatial.distance import cosine, correlation
from src.features.utils import load_features

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import numpy as np
import logging
import pickle
import click
import glob


def train_model(data_x: str, metric: callable, n_clusters=61, model_type="medoids"):
    """ Train model_type on data_x using metric. """
    metrics_lambdas = {
        "1-norm": lambda x, y: np.linalg.norm(x - y, ord=1),
        "2-norm": lambda x, y: np.linalg.norm(x - y, ord=2),
        "cosine": lambda x, y: cosine(x, y),
        "correlation": lambda x, y: correlation(x, y),
    }

    if model_type == "medoids":
        model = KMedoids(n_clusters=n_clusters, metric=metrics_lambdas[metric]).fit(data_x)
    elif model_type == "means":
        model = KMeans(n_clusters=n_clusters).fit(data_x)
    return model


@click.command()
@click.argument('feature_filepath', type=click.Path(exists=True))
@click.argument('model_type', type=str)
@click.argument('metric', type=str)
def main(feature_filepath, model_type, metric):
    data_x = load_features(feature_filepath)

    model = train_model(data_x, metric, n_clusters=61, model_type=model_type)

    print(f"Model inertia on 61 clusters: {model.inertia_}")

    if model_type == "means":
        with open(f"models/trained_model.pkl", 'wb') as f:
            pickle.dump(model, f)

    return model


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
