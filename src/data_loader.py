""" Loading MNIST dataset from IDX files"""

from pathlib import Path 

import numpy as np 

DATA_DIR = Path(__file__).resolve().parent.parent/"data"/"raw"

def read_idx(path):
    """Read an unsigned byte IDX file and return it as a numpy array"""
    with open(path,"rb") as f:
        magic = f.read(4)
        if magic[0] != 0 or magic[1] != 0 or magic[2] != 0x08:
            raise ValueError(f"{path} is not an unsigned-byte IDX file")

        n_dims = magic[3]
        shape = np.frombuffer(f.read(4 * n_dims), dtype=">u4")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if data.size != np.prod(shape):
        raise ValueError(
            f"{path}: Header says {np.prod(shape)} values file has {data.size} values"
        )
    return data.reshape(shape)

def load_mnist(data_dir=DATA_DIR):
    """Return the train_images, train_labels, test_images and test_labels """
    train_images = read_idx(data_dir/"train-images.idx3-ubyte")
    train_labels = read_idx(data_dir/"train-labels.idx1-ubyte")
    test_images = read_idx(data_dir/"t10k-images.idx3-ubyte")
    test_labels = read_idx(data_dir/"t10k-labels.idx1-ubyte")

    return (train_images, train_labels), (test_images, test_labels)

def preprocess_images(images):
    flat = images.reshape(images.shape[0],-1)
    return flat.astype(np.float32)/255

def one_hot(labels,num_classes=10):
    return np.eye(num_classes, dtype=np.float32)[labels]

def train_val_split(x,y,val_size=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])
