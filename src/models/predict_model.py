# Written by Francesco Paissan
from src.models.train_model import train_model
from src.features.utils import load_features
from src.models.utils import load_model

import numpy as np
import logging
import click
import os


def label_cluster(x: np.array, y_dict: np.array, model):
    """
    Assign label to every cluster detected
    """
    y_pred = model.predict(x)

    print(y_pred)

@click.command()
@click.argument('features_filepath', type=click.Path(exists=True))
@click.argument('model_type', type=str)
def main(features_filepath, model_type):
    model = load_model(model_type)

    data_x = load_features(os.path.join(features_filepath, "labeled"))
    label_cluster(data_x, None, model)

    for type in ['unlabeled', 'labeled']:
        data_x = load_features(os.path.join(features_filepath, type))
        y_pred = model.predict(data_x)

        print(f"In the {type} samples there are {len(np.unique(y_pred))} unique labeles")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
