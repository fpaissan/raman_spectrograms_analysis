# Written by Francesco Paissan
from src.models.train_model import train_model
from src.features.utils import load_features
import numpy as np
import logging
import click
import os

from src.models.utils import load_model


@click.command()
@click.argument('features_filepath', type=click.Path(exists=True))
@click.argument('model_type', type=str)
def main(features_filepath, model_type):
    model = load_model(model_type)

    for type in ['unlabeled', 'labeled']:
        data_x = load_features(os.path.join(features_filepath, type))
        y_pred = model.predict(data_x)

        print(f"In the {type} samples there are {len(np.unique(y_pred))} unique labeles")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
