# Written by Francesco Paissan
from progress.bar import ShadyBar
import pickle
import glob


def save_feat_files(feat, path):
    with open(path, "wb") as f:
        pickle.dump(feat, f)


def load_features(path):
    file_list = glob.glob('{0}/*'.format(path))

    with ShadyBar(f"Loading dataset...", max=len(file_list)) as bar:
        for f in file_list:
            with open(f, "rb") as f:
                data_x = pickle.load(f)

            bar.next()

    return data_x