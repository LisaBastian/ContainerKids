import torch
import argparse
from config import (
    DEVICE, EMBEDDING_SIZES, SEED, TEST_SIZE, GRAPH_FILE
)
from data_utils import set_seed, load_graphs, split_graphs
from train import run_cv, train_final, evaluate
from perm_test import permutation_test
from explain import explain_graphs
from visualize import plot_confusion, plot_uv_histogram
from model import GCNWithEdgeWeights


def main():
    parser = argparse.ArgumentParser(description="Graph classification pipeline")
    parser.add_argument(
        "--mode",
        choices=["cv", "test", "perm", "explain"],
        required=True,
        help="Run the stages in the displayed order:"
    )
    parser.add_argument(
        "--emb",
        type=int,
        default=EMBEDDING_SIZES[0],
        help="Embedding size to use"
    )
    args = parser.parse_args()
    set_seed(SEED)

    graphs = load_graphs(GRAPH_FILE)
    tv_g, tv_lbl, test_g, test_lbl = split_graphs(graphs, TEST_SIZE, SEED)

    if args.mode == "cv":
        mean_acc, std_acc, mean_cm = run_cv(tv_g, tv_lbl, DEVICE, args.emb)
        plot_confusion(mean_cm, f"CV Confusion (emb={args.emb})")
        print(f"CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    elif args.mode == "test":
        model = train_final(tv_g, DEVICE, args.emb)
        torch.save(model.state_dict(), f"final_model_emb{args.emb}.pt")

    elif args.mode == "perm":
        model = GCNWithEdgeWeights(
            NODE_FEAT_DIM, EDGE_FEAT_DIM, GRAPH_FEAT_DIM, args.emb
        ).to(DEVICE)
        model.load_state_dict(torch.load(f"final_model_emb{args.emb}.pt"))
        accr = evaluate(get_loader(test_g, BATCH_SIZE, shuffle=False), model, DEVICE)[0]
        results, pval = permutation_test(model, test_g, test_lbl, accr, DEVICE)
        print(f"Permutation p-value: {pval:.4f}")

    elif args.mode == "explain":
        model = GCNWithEdgeWeights(
            NODE_FEAT_DIM, EDGE_FEAT_DIM, GRAPH_FEAT_DIM, args.emb
        ).to(DEVICE)
        model.load_state_dict(torch.load(f"final_model_emb{args.emb}.pt"))
        results = explain_graphs(model, graphs)
        plot_uv_histogram(results)

    else:
        parser.error("Unknown mode")


if __name__ == "__main__":
    main()
# %%
