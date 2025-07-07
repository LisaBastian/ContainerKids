import copy
import numpy as np
import torch
from joblib import Parallel, delayed
from train import evaluate
from data_utils import get_loader
from config import N_PERMUTATIONS, SEED, BATCH_SIZE


def permutation_test(model, test_graphs, test_labels, observed_acc, device):
    np.random.seed(999)
    def single_perm(i):
        shuffled = np.random.permutation(test_labels)
        perm_g = []
        for g, y in zip(test_graphs, shuffled):
            g2 = copy.deepcopy(g)
            g2.y = torch.tensor([y], dtype=g.y.dtype)
            perm_g.append(g2)
        loader = get_loader(perm_g, BATCH_SIZE, shuffle=False)
        acc, _ = evaluate(loader, model, device)
        return acc

    results = Parallel(n_jobs=-1)(delayed(single_perm)(i) for i in range(N_PERMUTATIONS))
    p_val = (np.sum(np.array(results) >= observed_acc) + 1) / (N_PERMUTATIONS + 1)
    return results, p_val