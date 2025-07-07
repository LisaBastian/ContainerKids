import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn.utils import clip_grad_norm_
from model import GCNWithEdgeWeights
from data_utils import get_loader, get_cv_splits
from config import (
    NODE_FEAT_DIM, EDGE_FEAT_DIM, GRAPH_FEAT_DIM,
    LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, LR_FACTOR, MIN_LR,
    BATCH_SIZE, CV_SPLITS, FINAL_EPOCHS, SEED,
)


def train_epoch(loader, model, optimizer, loss_fn, device, max_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _ = model(
            batch.x.float(),
            batch.edge_index,
            batch.edge_attr.view(-1,1),
            batch.batch,
            batch.graph_features.float(),
        )
        loss = loss_fn(out, batch.y.float())
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(loader, model, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(
                batch.x.float(),
                batch.edge_index,
                batch.edge_attr.view(-1,1),
                batch.batch,
                batch.graph_features.float(),
            )
            preds = (torch.sigmoid(out) > 0.5).long()
            ys.extend(batch.y.tolist())
            ps.extend(preds.tolist())
    acc = accuracy_score(ys, ps)
    cm  = confusion_matrix(ys, ps)
    return acc, cm


def run_cv(graphs, labels, device, embedding_size: int):
    fold_accs = []
    cum_cm = np.zeros((2,2), dtype=int)
    splits = get_cv_splits(graphs, labels, n_splits=CV_SPLITS, seed=SEED)
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        train_g = [graphs[i] for i in train_idx]
        val_g   = [graphs[i] for i in val_idx]
        tr_loader = get_loader(train_g, BATCH_SIZE, shuffle=True)
        vl_loader = get_loader(val_g,   BATCH_SIZE, shuffle=False)

        model = GCNWithEdgeWeights(
            NODE_FEAT_DIM, EDGE_FEAT_DIM,
            GRAPH_FEAT_DIM, embedding_size
        ).to(device)
        opt = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=LR_FACTOR,
            patience=PATIENCE, min_lr=MIN_LR,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_val = float('inf')
        no_imp   = 0
        ckpt     = f"best_fold{fold_idx}.pt"

        for epoch in range(1, MAX_EPOCHS+1):
            train_epoch(tr_loader, model, opt, loss_fn, device)
            val_acc, _ = evaluate(vl_loader, model, device)
            val_loss = 1 - val_acc
            sched.step(val_loss)
            if val_loss < best_val:
                best_val, no_imp = val_loss, 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_imp += 1
                if no_imp >= PATIENCE:
                    break

        model.load_state_dict(torch.load(ckpt))
        acc, cm = evaluate(vl_loader, model, device)
        fold_accs.append(acc)
        cum_cm += cm

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    mean_cm  = cum_cm / CV_SPLITS
    return mean_acc, std_acc, mean_cm


def train_final(graphs, device, embedding_size: int):
    loader = get_loader(graphs, BATCH_SIZE, shuffle=True)
    model  = GCNWithEdgeWeights(
        NODE_FEAT_DIM, EDGE_FEAT_DIM,
        GRAPH_FEAT_DIM, embedding_size
    ).to(device)
    opt    = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(FINAL_EPOCHS):
        train_epoch(loader, model, opt, loss_fn, device)
    return model