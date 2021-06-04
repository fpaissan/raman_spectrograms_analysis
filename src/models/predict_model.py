import glob
import logging
import pickle

import click
import numpy as np
from progress.bar import ShadyBar

from src.models.utils import load_data


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('features_filepath', type=click.Path(exists=True))
def main(model_filepath, features_filepath):
    file_list = glob.glob('{0}/*'.format(features_filepath))

    r_start = 0
    r_end = 1400

    model = pickle.load(open(model_filepath, 'rb'))
    # logging.log("Model loaded", logging.INFO)

    # TODO: Check if 1D is ok
    data_x = list()
    data_y = list()
    with ShadyBar(f"Loading dataset...", max=len(file_list)) as bar:
        for f in file_list:
            # print(load_data(f, r_start, r_end).shape, f.split('/')[-1].split('.')[0])
            data_x.append(load_data(f, r_start, r_end))
            data_y.append(f.split('/')[-1].split('.')[0])
            bar.next()

    data_x = np.array(data_x)
    print(data_x.shape)
    y_pred = model.predict(data_x)

    print(len(np.unique(y_pred)))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
