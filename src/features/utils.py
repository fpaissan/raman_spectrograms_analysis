# Written by Francesco Paissan
import pickle

def save_feat_files(feat, path):
    with open(path, "wb") as f:
        pickle.dump(feat, f)