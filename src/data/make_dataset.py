# -*- coding: utf-8 -*-
import pandas as pd

from src.data.utils import read_data, normalize_col

from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from progress.bar import ShadyBar
import logging
import click
import glob
import os


def make_unlabeled(input_filepath, output_filepath, norm_type="integral"):
    file_list = glob.glob(input_filepath + '/*')
    for f in file_list:
        df = read_data(f)
        wl = df.wl
        data = df.drop(columns=['wl'])

        with ShadyBar(f"Processing file... {f}", max=len(data.columns)) as bar:
            for col in data:
                norm_col = normalize_col(wl, data[col], norm_type=norm_type)
                norm_col.to_csv(
                    os.path.join(output_filepath, f"{'_'.join(f.split('/')[-1].split('_')[:-1])}_{col}.csv"))

                bar.next()


def make_labeled(input_filepath, output_filepath, norm_type="integral"):
    file_list = glob.glob(input_filepath + '/*')
    with ShadyBar(f"Processing labeled files...", max=len(file_list)) as bar:
        for f in file_list:
            df = pd.read_csv(f, delim_whitespace=True, names=['wl', 'ri'])
            wl = df.wl
            data = df.drop(columns=['wl'])

            for col in data:
                norm_col = normalize_col(wl, data[col], norm_type=norm_type)
                norm_col.to_csv(
                    os.path.join(output_filepath, f"{f.split('/')[-1]}.csv"))

            bar.next()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    make_unlabeled(os.path.join(input_filepath, "unlabeled"), os.path.join(output_filepath, "unlabeled"))
    make_labeled(os.path.join(input_filepath, "labeled"), os.path.join(output_filepath, "labeled"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
