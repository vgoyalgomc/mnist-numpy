# MNIST from scratch with NumPy

## Goal

This repository explores how to classify handwritten digits from the MNIST dataset using NumPy. The objective is to build the core machine-learning logic from scratch, without relying on machine-learning libraries, and to make the steps behind digit classification easier to understand.

## Data

The MNIST dataset is included in this repository under `data/raw/`, so no separate download is needed. The files are in the original IDX binary format:

| File | Contents |
|------|----------|
| `train-images.idx3-ubyte` | 60,000 training images (28 × 28 grayscale) |
| `train-labels.idx1-ubyte` | 60,000 training labels (digits 0–9) |
| `t10k-images.idx3-ubyte` | 10,000 test images (28 × 28 grayscale) |
| `t10k-labels.idx1-ubyte` | 10,000 test labels (digits 0–9) |

**Credit:** The MNIST database was created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges, and is made available under the [Creative Commons Attribution-Share Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.

## Setup

Requires Python 3.12 or newer.

Clone the repository and move into it:

```bash
git clone https://github.com/vgoyalgomc/mnist-numpy.git
cd mnist-numpy
```

Create and activate a virtual environment, then install the project dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
```

`python --version` should print `Python 3.12.x` before you install anything.

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\activate` instead.
