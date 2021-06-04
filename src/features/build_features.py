# -*- coding: utf-8 -*-
import pandas as pd

from src.data.utils import read_data, normalize_col

from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from progress.bar import ShadyBar
import numpy as np
import logging
import click
import glob
import os


def find_maxpeak_argmax(input_filepath, output_filepath):
    file_list = glob.glob(input_filepath + '/*')
    features_set = []
    with ShadyBar(f"Extracting features {input_filepath}...", max=len(file_list)) as bar:
        for f in file_list:
            interim_data = np.loadtxt(f, delimiter=',', skiprows=1)
            features_set.append(np.argmax(interim_data[:, 1]))

            bar.next()

    np.savetxt(os.path.join(output_filepath, "peaks_features.csv"), np.array(features_set))


@click.command()
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(features_path, output_filepath):
    """
        Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    for type in ['labeled', 'unlabeled']:
        find_maxpeak_argmax(os.path.join(features_path, type), os.path.join(output_filepath, type))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
