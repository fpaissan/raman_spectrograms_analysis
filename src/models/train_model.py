# Written by Francesco Paissan
from scipy.spatial.distance import cosine, correlation
from sklearn_extra.cluster import KMedoids
from progress.bar import ShadyBar
import numpy as np
import logging
import pickle
import click
import glob


def train_model(data_x, metric, n_clusters=60):
    """ Train model_type on data_x using metric. """
    metrics_lambdas = {
        "1-norm": lambda x, y: np.linalg.norm(x - y, ord=1),
        "2-norm": lambda x, y: np.linalg.norm(x - y, ord=2),
        "cosine": lambda x, y: cosine(x, y),
        "correlation": lambda x, y: correlation(x, y),
    }

    k_mean = KMedoids(n_clusters=n_clusters, metric=metrics_lambdas[metric])
    k_mean.fit(data_x)

    return k_mean


@click.command()
@click.argument('feature_filepath', type=click.Path(exists=True))
@click.argument('metric', type=str)
def main(feature_filepath, metric):
    file_list = glob.glob('{0}/*'.format(feature_filepath))

    with ShadyBar(f"Loading dataset...", max=len(file_list)) as bar:
        for f in file_list:
            with open(f, "rb") as f:
                data_x = pickle.load(f)

            bar.next()

    k_mean = train_model(data_x, metric)

    with open(f"models/trained_model.pkl", 'wb') as f:
        pickle.dump(k_mean, f)

    return k_mean


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
