"""
vlstm.py
P2-ETF-VLSTM-SIGNAL

VSN + LSTM architecture following Saly-Kaufmann et al. (Oxford, 2026).
Adapted for ETF classification (next-day best ETF selection).

Architecture:
  Input [batch, lookback, n_features]
    → VSN: per-timestep dynamic feature gating
    → LSTM: 2-layer recurrent encoder
    → Linear projection head → softmax → ETF probabilities

Two loss modes:
  "ce"     : cross-entropy on argmax ETF label
  "sharpe" : end-to-end Sharpe ratio optimisation on portfolio returns
             (model output = position weights, loss = -annualised Sharpe)

Validation always uses Sharpe ratio to pick the winner between the two modes.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Variable Selection Network ────────────────────────────────────────────────

class VSN(nn.Module):
    """
    Variable Selection Network (VSN) from Temporal Fusion Transformer.
    Applies per-feature nonlinear embedding then dynamic soft gating.

    At each timestep t:
      1. Each feature i → independent Linear → ELU embedding h_{t,i}
      2. All embeddings concatenated → gating network → softmax weights α_t
      3. Output = weighted sum of embeddings

    Input:  [batch, seq_len, n_features]
    Output: [batch, seq_len, hidden_dim]
            [batch, seq_len, n_features]  ← attention weights (for logging)
    """

    def __init__(self, n_features: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Per-feature embeddings (shared weights across time)
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ELU(),
            )
            for _ in range(n_features)
        ])

        # Gating network: concat of all embeddings → softmax weights
        self.gate = nn.Sequential(
            nn.Linear(n_features * hidden_dim, n_features),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, n_features]
        """
        B, T, C = x.shape

        # Embed each feature independently
        embedded = []
        for i, emb in enumerate(self.feature_embeddings):
            # x[..., i:i+1] → [B, T, 1] → [B, T, hidden_dim]
            embedded.append(emb(x[..., i:i+1]))

        # Stack → [B, T, n_features, hidden_dim]
        stacked = torch.stack(embedded, dim=2)

        # Gating: flatten embeddings → [B, T, n_features * hidden_dim]
        flat = stacked.reshape(B, T, -1)
        weights = F.softmax(self.gate(flat), dim=-1)          # [B, T, n_features]

        # Weighted sum of embeddings
        # weights: [B, T, n_features, 1] × stacked: [B, T, n_features, hidden_dim]
        weights_exp = weights.unsqueeze(-1)                    # [B, T, C, 1]
        out = (weights_exp * stacked).sum(dim=2)               # [B, T, hidden_dim]
        out = self.dropout(out)

        return out, weights


# ── VLSTM Model ───────────────────────────────────────────────────────────────

class VLSTM(nn.Module):
    """
    VSN + LSTM classifier for next-day ETF selection.

    Forward pass returns:
      logits   : [batch, n_etfs]   raw scores (before softmax)
      attn     : [batch, seq_len, n_features]  VSN attention weights
    """

    def __init__(
        self,
        n_features:  int,
        n_etfs:      int,
        hidden_dim:  int   = 128,
        lstm_layers: int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.vsn  = VSN(n_features, hidden_dim, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size   = hidden_dim,
            hidden_size  = hidden_dim,
            num_layers   = lstm_layers,
            batch_first  = True,
            dropout      = dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_etfs),
        )

    def forward(self, x: torch.Tensor):
        vsn_out, attn = self.vsn(x)                   # [B, T, hidden], [B, T, C]
        lstm_out, _   = self.lstm(vsn_out)            # [B, T, hidden]
        h_T           = lstm_out[:, -1, :]            # [B, hidden]  final state
        logits        = self.head(h_T)                # [B, n_etfs]
        return logits, attn


# ── Sharpe ratio loss ─────────────────────────────────────────────────────────

def sharpe_loss(
    logits:    torch.Tensor,    # [B, n_etfs]
    y_returns: torch.Tensor,    # [B, n_etfs]  next-day raw returns
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    End-to-end differentiable Sharpe ratio loss.
    Position weights = softmax(logits) — long-only, sum to 1.
    Portfolio return = sum(weights * next_day_returns).
    Loss = -annualised Sharpe ratio over the batch.
    """
    weights    = F.softmax(logits, dim=-1)             # [B, n_etfs]
    port_ret   = (weights * y_returns).sum(dim=-1)     # [B]

    mean_ret   = port_ret.mean()
    std_ret    = port_ret.std() + eps
    sharpe     = mean_ret / std_ret * math.sqrt(252)

    return -sharpe


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(
    X:        np.ndarray,
    y_labels: np.ndarray,
    y_returns:np.ndarray,
    lookback: int,
) -> tuple:
    """
    Build overlapping sequences of length `lookback`.
    Returns arrays indexed from lookback onwards.

    X_seq[i]   = X[i : i+lookback]          features over window
    y_lab[i]   = y_labels[i+lookback]       label at end of window
    y_ret[i]   = y_returns[i+lookback]      returns at end of window
    """
    n      = len(X)
    n_seq  = n - lookback

    X_seq  = np.stack([X[i:i+lookback] for i in range(n_seq)], axis=0)
    y_lab  = y_labels[lookback:]
    y_ret  = y_returns[lookback:]

    return X_seq, y_lab, y_ret


# ── Feature scaler ────────────────────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple:
    """
    Standardise features using train-set mean/std (no lookahead).
    Returns scaled arrays + (mean, std) for inference.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    return (
        (X_train - mean) / std,
        (X_val   - mean) / std,
        (X_test  - mean) / std,
        mean, std,
    )


def scale_single(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale a single sample using pre-fitted mean/std."""
    return (X - mean) / std


# ── Lookback auto-selector ────────────────────────────────────────────────────

def find_best_lookback(
    X:         np.ndarray,
    y_labels:  np.ndarray,
    y_returns: np.ndarray,
    train_pct: float,
    val_pct:   float,
    n_etfs:    int,
    n_features:int,
    candidates: list = [30, 45, 60],
    epochs:    int   = 20,          # quick search — fewer epochs
) -> int:
    """
    Train a small VLSTM for a few epochs with each candidate lookback.
    Returns the lookback with best validation Sharpe.
    """
    set_seed(42)
    device = torch.device("cpu")
    best_lb, best_sharpe = candidates[0], -999.0

    for lb in candidates:
        try:
            X_seq, y_lab, y_ret = build_sequences(X, y_labels, y_returns, lb)
            n      = len(X_seq)
            t_end  = int(n * train_pct)
            v_end  = int(n * (train_pct + val_pct))

            if t_end < 50 or (v_end - t_end) < 20:
                continue

            Xtr, Xva = X_seq[:t_end], X_seq[t_end:v_end]
            ltr, lva = y_lab[:t_end], y_lab[t_end:v_end]
            rtr, rva = y_ret[:t_end], y_ret[t_end:v_end]

            Xtr_s, Xva_s, _, _, _ = scale_features(Xtr, Xva, Xva)

            model = VLSTM(n_features, n_etfs, hidden_dim=64, lstm_layers=1).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

            ds    = TensorDataset(
                torch.tensor(Xtr_s, dtype=torch.float32),
                torch.tensor(ltr,   dtype=torch.long),
                torch.tensor(rtr,   dtype=torch.float32),
            )
            loader = DataLoader(ds, batch_size=64, shuffle=True)

            model.train()
            for _ in range(epochs):
                for xb, lb_, rb in loader:
                    logits, _ = model(xb)
                    loss = F.cross_entropy(logits, lb_)
                    opt.zero_grad(); loss.backward(); opt.step()

            # Evaluate on val set
            model.eval()
            with torch.no_grad():
                xv     = torch.tensor(Xva_s, dtype=torch.float32)
                rv     = torch.tensor(rva,   dtype=torch.float32)
                logits, _ = model(xv)
                w      = F.softmax(logits, dim=-1)
                pr     = (w * rv).sum(dim=-1).numpy()
                sh     = pr.mean() / (pr.std() + 1e-8) * math.sqrt(252)

            if sh > best_sharpe:
                best_sharpe = sh
                best_lb     = lb

        except Exception:
            continue

    return best_lb


# ── Core trainer ──────────────────────────────────────────────────────────────

def train_vlstm(
    X_train:   np.ndarray,       # [T_train, lookback, n_features]
    y_train_l: np.ndarray,       # [T_train]  labels
    y_train_r: np.ndarray,       # [T_train, n_etfs] returns
    X_val:     np.ndarray,
    y_val_l:   np.ndarray,
    y_val_r:   np.ndarray,
    n_etfs:    int,
    loss_mode: str   = "ce",     # "ce" or "sharpe"
    hidden_dim:int   = 128,
    lstm_layers:int  = 2,
    dropout:   float = 0.3,
    lr:        float = 5e-4,
    epochs:    int   = 80,
    batch_size:int   = 64,
    patience:  int   = 15,
    seed:      int   = 42,
) -> dict:
    """
    Train a VLSTM model.

    Returns dict with:
        model        : trained VLSTM (in eval mode)
        val_sharpe   : validation Sharpe ratio
        val_ce       : validation cross-entropy loss
        scale_mean   : feature mean for inference scaling
        scale_std    : feature std  for inference scaling
        train_time_s : wall-clock training time in seconds
        loss_mode    : which loss was used
    """
    set_seed(seed)
    device     = torch.device("cpu")
    t_start    = time.time()

    n_features = X_train.shape[2]

    model      = VLSTM(n_features, n_etfs, hidden_dim, lstm_layers, dropout).to(device)
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.tensor(X_train,   dtype=torch.float32),
        torch.tensor(y_train_l, dtype=torch.long),
        torch.tensor(y_train_r, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)

    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_val_l_t = torch.tensor(y_val_l, dtype=torch.long).to(device)
    y_val_r_t = torch.tensor(y_val_r, dtype=torch.float32).to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    history        = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for xb, lb, rb in train_loader:
            xb, lb, rb = xb.to(device), lb.to(device), rb.to(device)
            logits, _  = model(xb)

            if loss_mode == "ce":
                loss = F.cross_entropy(logits, lb)
            else:   # sharpe
                loss = sharpe_loss(logits, rb)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val_t)

            if loss_mode == "ce":
                val_loss = F.cross_entropy(val_logits, y_val_l_t).item()
            else:
                val_loss = sharpe_loss(val_logits, y_val_r_t).item()

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # ── Load best checkpoint ──────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ── Compute val Sharpe and val CE for reporting ───────────────────────────
    with torch.no_grad():
        val_logits, _ = model(X_val_t)
        val_ce        = F.cross_entropy(val_logits, y_val_l_t).item()
        w             = F.softmax(val_logits, dim=-1)
        port_ret      = (w * y_val_r_t).sum(dim=-1).cpu().numpy()
        val_sharpe    = (port_ret.mean() / (port_ret.std() + 1e-8)
                         * math.sqrt(252))

    train_time = time.time() - t_start

    return {
        "model":         model,
        "val_sharpe":    float(val_sharpe),
        "val_ce":        float(val_ce),
        "best_val_loss": float(best_val_loss),
        "train_time_s":  float(train_time),
        "loss_mode":     loss_mode,
        "history":       history,
        "epochs_run":    len(history),
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    model:     VLSTM,
    X:         np.ndarray,         # [T, lookback, n_features] already scaled
    batch_size:int = 256,
) -> tuple:
    """
    Run inference on scaled sequences.

    Returns:
        preds  : np.ndarray [T]        argmax ETF index
        proba  : np.ndarray [T, n_etfs] softmax probabilities
        attn   : np.ndarray [T, lookback, n_features] VSN weights
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_proba, all_attn = [], [], []
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (xb,) in loader:
            xb         = xb.to(device)
            logits, at = model(xb)
            prob       = F.softmax(logits, dim=-1)
            all_preds.append(prob.argmax(dim=-1).cpu().numpy())
            all_proba.append(prob.cpu().numpy())
            all_attn.append(at.cpu().numpy())

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_proba, axis=0),
        np.concatenate(all_attn,  axis=0),
    )


# ── VSN attention summary ─────────────────────────────────────────────────────

def top_vsn_features(
    attn:          np.ndarray,     # [T, lookback, n_features]
    feature_names: list,
    top_k:         int = 5,
) -> list:
    """
    Average VSN attention weights over time and lookback.
    Returns top-k feature names with their mean attention weight.
    """
    mean_attn = attn.mean(axis=(0, 1))                 # [n_features]
    idx       = np.argsort(mean_attn)[::-1][:top_k]
    return [
        {"feature": feature_names[i], "weight": float(mean_attn[i])}
        for i in idx
    ]
