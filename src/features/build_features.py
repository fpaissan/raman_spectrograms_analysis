# -*- coding: utf-8 -*-
from src.features.peak_features import find_maxpeak_argmax, find_maxpeak_2d, gen_n_peak
from src.features.engineered_features import extract_features

from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import logging
import click
import os


@click.command()
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('type_feature', type=str)
def main(features_path, output_filepath, type_feature):
    """
        Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    feat_extractor = {
        'maxpeak_argmax': find_maxpeak_argmax,
        'maxpeak_2d': find_maxpeak_2d,
        'maxpeak_n_argmax': gen_n_peak(10, "argmax"),
        'maxpeak_n_2d': gen_n_peak(5, "amp"),
        'eng': extract_features
    }

    for type in ['labeled', 'unlabeled']:
        feat_extractor[type_feature](os.path.join(features_path, type), os.path.join(output_filepath, type))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
