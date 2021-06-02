import pickle

from sklearn.cluster import KMeans
from progress.bar import ShadyBar
import numpy as np
import logging
import click
import glob

from src.models.utils import load_data


@click.command()
@click.argument('feature_filepath', type=click.Path(exists=True))
def main(feature_filepath):
    file_list = glob.glob('{0}/*'.format(feature_filepath))

    r_start = 0
    r_end = 1400

    # TODO: Check if 1D is ok
    data_x = list()
    with ShadyBar(f"Loading dataset...", max=len(file_list)) as bar:
        for f in file_list:
            if not "bkg" in f:
                data_x.append(load_data(f, r_start, r_end))
            bar.next()

    data_x = np.array(data_x)
    k_mean = KMeans(n_clusters=61)
    k_mean.fit(data_x)

    with open(f"models/k_means_range{r_start}-{r_end}.pkl", 'wb') as f:
        pickle.dump(k_mean, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
