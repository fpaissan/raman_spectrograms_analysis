#Written by Francesco Paissan
from src.models.train_model import train_model
from src.features.utils import load_features
import numpy as np
import logging
import click
import os


@click.command()
@click.argument('features_filepath', type=click.Path(exists=True))
def main(features_filepath):
    for type in ['unlabeled', 'labeled']:
        data_x = load_features(os.path.join(features_filepath, type))
        model = train_model(data_x, "2-norm", n_clusters=61)

        y_pred = model.predict(data_x)

        print(f"In the {type} samples there are {len(np.unique(y_pred))} unique labeles")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
