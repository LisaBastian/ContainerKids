from pathlib import Path
import torch

# === Paths ===
ROOT_DIR       = Path(__file__).parent
GRAPH_FILE     = ROOT_DIR / "encoding_graphs.pt"
EXPLANATION_DIR = ROOT_DIR / "gnn_explanations"

# === Model hyperparameters ===
NODE_FEAT_DIM   = 26
EDGE_FEAT_DIM   = 1
GRAPH_FEAT_DIM  = 25
EMBEDDING_SIZES = [8] #best embedding size for encoding based on CV for: 2, 4, 8, 16, 32

# === Training hyperparameters ===
CV_SPLITS      = 10
TEST_SIZE      = 0.10
BATCH_SIZE     = 64
SEED           = 42
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 0.01
MAX_EPOCHS     = 4000
PATIENCE       = 100
LR_FACTOR      = 0.75
MIN_LR         = 1e-6
FINAL_EPOCHS   = 3000
N_PERMUTATIONS = 5000

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")