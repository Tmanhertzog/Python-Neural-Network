# Tristan Hertzog 2025, Jaxon Powell 2025

"""
System Arguments:
[1] mode ∈ {C, R}       # regression or classification
[2] D ∈ ℕ               # number of dimensions (feature dimension)
[3] C ∈ ℕ               # output vector dimension, y ∈ ℝ^C
[4] train_N ∈ ℕ         # number of lines in training set (number of training examples)
[5] dev_N ∈ ℕ           # number of lines in dev set 
[6] std

Ex: python make_data.py C 5 4 1000 200 0.5
"""

from pathlib import Path
import argparse
from typing import Tuple

import numpy as np

#=========================
# Updating TXT Files
#=========================

def count_lines(file_path):
    if not file_path.exists():
        return 0
    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def log_data_info(folder, message):
    folder.mkdir(parents=True, exist_ok=True)
    info_path = folder / "data_info.txt"
    idx = count_lines(info_path) + 1
    with info_path.open("a", encoding="utf-8") as f:
        f.write(f"dataset{idx} - {message}\n")


# =========================
# TXT saving
# =========================

def save_matrix_txt(X: np.ndarray, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    is_int = np.issubdtype(X.dtype, np.integer)

    with filename.open("w", encoding="utf-8") as f:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if is_int:
                    f.write(f"{int(X[i, j])} ")
                else:
                    f.write(f"{X[i, j]:.5f} ")
            f.write("\n")


# =========================
# Data generation
# =========================

def make_linear_generator(rng: np.random.Generator, D: int, C: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a ground-truth linear mapping:
        y = xW + b
    where W is (D, C) and b is (C,)
    """
    W = rng.standard_normal((D, C))
    b = rng.standard_normal((C,))
    return W, b


def generate_regression(rng: np.random.Generator, D: int, C: int, N: int, std: float, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x ~ N(0,1)^D
    y = xW + b + noise, noise ~ N(0, std^2)^C
    """
    x = rng.standard_normal((N, D))
    noise = rng.standard_normal((N, C)) * std
    y = x @ W + b + noise
    return x, y


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)


def generate_classification_labels(rng: np.random.Generator, D: int, C: int, N: int, std: float, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x ~ N(0,1)^D
    logits = xW + b + noise
    y ∈ {0, 1, ..., C-1}  (integer class labels)
    """
    x = rng.standard_normal((N, D))
    logits = x @ W + b + rng.standard_normal((N, C)) * std

    y = np.argmax(logits, axis=1)  # shape (N,)
    return x, y

#=========================
# Creating Data Files
#=========================

def create_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["C", "R"])
    parser.add_argument("D", type=int)
    parser.add_argument("C", type=int)
    parser.add_argument("train_N", type=int)
    parser.add_argument("dev_N", type=int)
    parser.add_argument("std", type=float)

    args = parser.parse_args()

    if args.D <= 0:
        parser.error("D must be positive int")
    if args.C <= 0:
        parser.error("C must be positive int")
    if args.train_N <= 0:
        parser.error("train_N must be positive int")
    if args.dev_N <= 0:
        parser.error("dev_N must be positive int")
    if args.std < 0:
        parser.error("std must be >= 0")

    type = args.mode
    D = args.D
    C = args.C
    train_N = args.train_N
    dev_N = args.dev_N
    std = args.std
    rng = np.random.default_rng()


    W, b = make_linear_generator(rng, D, C)
    outdir = ""

    #------------------
    # Regression
    #------------------

    if type == "R":
        train_x, train_y = generate_regression(rng, D, C, train_N, std, W, b)
        dev_x, dev_y = generate_regression(rng, D, C, dev_N, std, W, b)
        outdir = Path("data/regression")
    

    #------------------
    # Classification
    #------------------

    else:
        if C < 2:
            parser.error("For classification, C (num classes) must be >= 2")

        train_x, train_y = generate_classification_labels(rng, D, C, train_N, std, W, b)
        dev_x, dev_y = generate_classification_labels(rng, D, C, dev_N, std, W, b)
        outdir = Path("data/classification")

    info_path = outdir / "data_info.txt"
    dataset_idx = count_lines(info_path) + 1
    
    # Write files
    save_matrix_txt(train_x, outdir / f"dataset{dataset_idx}.train_x.txt")
    save_matrix_txt(train_y, outdir / f"dataset{dataset_idx}.train_y.txt")
    save_matrix_txt(dev_x,   outdir / f"dataset{dataset_idx}.dev_x.txt")
    save_matrix_txt(dev_y,   outdir / f"dataset{dataset_idx}.dev_y.txt")

    # Log
    log_data_info(outdir, f"mode={type}, D={D}, C={C}, train_N={train_N}, dev_N={dev_N}, std={std}")

    print(f"Wrote dataset to: {outdir}")

#=========================
# Program
#=========================

if __name__ == "__main__":
    create_data()