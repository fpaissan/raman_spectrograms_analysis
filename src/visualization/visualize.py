# Written by Francesco Paissan
from src.features.utils import load_features
import src.visualization.vis_func as vf

import matplotlib.pyplot as plt
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
        plt.close()
        if type == 'unlabeled':
            vf.cluster_vis(f"data/interim/{type}", y_pred)
            plt.close()

            S1_pred = y_pred[:121]
            S2_pred = y_pred[121:]

            vf.proportion_per_cluster(data_x[:121], S1_pred, "S1")
            vf.proportion_per_cluster(data_x[121:], S2_pred, "S2")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
