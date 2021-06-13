# Written by Francesco Paissan
from src.models.train_model import train_model
from src.features.utils import load_features
import src.visualization.vis_func as vf

import logging
import click
import os

from src.models.utils import load_model


@click.command()
@click.argument('features_filepath', type=click.Path(exists=True))
def main(features_filepath):
    model = load_model()

    for type in ['unlabeled', 'labeled']:
        data_x = load_features(os.path.join(features_filepath, type))
        y_pred = model.predict(data_x)

        vf.proportion_per_cluster(data_x, y_pred, type)
        if type == 'unlabeled':
            vf.cluster_vis(f"data/interim/{type}", y_pred)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
