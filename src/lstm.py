"""Sequence model: single-layer LSTM classifier over SAX symbol windows.

Architecture fixed a priori (PLAN.md): one LSTM layer, hidden size 64,
symbol embedding 16, linear head to N classes. The only setting tuned is
the epoch count, via early stopping on the blocked validation fold(s)
inside the training period. Training windows may overlap (stride 4) as
data augmentation within the training period; evaluation windows are the
same non-overlapping windows used for the Markov classifier.

Determinism: torch.manual_seed(seed), deterministic algorithms enabled.
"""

import numpy as np
import torch
import torch.nn as nn

EMB = 16
HIDDEN = 64
LR = 1e-3
BATCH = 128
MAX_EPOCHS = 50
PATIENCE = 5
TRAIN_STRIDE = 4


class LSTMClassifier(nn.Module):
    def __init__(self, alphabet, n_classes):
        super().__init__()
        self.emb = nn.Embedding(alphabet, EMB)
        self.lstm = nn.LSTM(EMB, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, n_classes)

    def forward(self, x):
        h = self.emb(x)
        _, (hn, _) = self.lstm(h)
        return self.head(hn[-1])


def strided_windows(sym, length, stride):
    idx = np.arange(0, len(sym) - length + 1, stride)
    return np.stack([sym[i:i + length] for i in idx]) if len(idx) else np.empty((0, length), dtype=sym.dtype)


def make_dataset(seqs, length, stride):
    xs, ys = [], []
    for ci, sym in enumerate(seqs):
        w = strided_windows(sym, length, stride)
        if len(w):
            xs.append(w)
            ys.append(np.full(len(w), ci))
    return (torch.from_numpy(np.concatenate(xs).astype(np.int64)),
            torch.from_numpy(np.concatenate(ys).astype(np.int64)))


def train_lstm(train_seqs, val_seqs, alphabet, length, seed=42, verbose=False):
    """Train with early stopping on val accuracy; returns (model, best_val_acc)."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    xt, yt = make_dataset(train_seqs, length, TRAIN_STRIDE)
    xv, yv = make_dataset(val_seqs, length, length)  # non-overlapping val windows
    model = LSTMClassifier(alphabet, len(train_seqs))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
        acc = eval_lstm(model, xv, yv)
        if verbose:
            print(f"  epoch {epoch}: val acc {acc:.3f}", flush=True)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def eval_lstm(model, x, y):
    model.eval()
    preds = []
    for i in range(0, len(x), 512):
        preds.append(model(x[i:i + 512]).argmax(dim=1))
    return float((torch.cat(preds) == y).float().mean())


@torch.no_grad()
def predict_lstm(model, windows_np):
    model.eval()
    x = torch.from_numpy(windows_np.astype(np.int64))
    preds = []
    for i in range(0, len(x), 512):
        preds.append(model(x[i:i + 512]).argmax(dim=1))
    return torch.cat(preds).numpy()
