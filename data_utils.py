import torch
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.loader import DataLoader
from config import (GRAPH_FILE, TEST_SIZE, SEED,
                     CV_SPLITS, BATCH_SIZE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graphs(path: Path):
    graphs = torch.load(path)
    for g in graphs:
        g.num_nodes = g.x.size(0)
    return graphs


def split_graphs(graphs, test_size: float, seed: int):
    labels = [int(g.y.item()) for g in graphs]
    train_val_graphs, test_graphs, train_val_labels, test_labels = train_test_split(
        graphs, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    return train_val_graphs, train_val_labels, test_graphs, test_labels


def get_cv_splits(graphs, labels, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(graphs, labels)


def get_loader(graphs, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )