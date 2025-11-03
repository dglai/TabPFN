import random
from dataclasses import dataclass
from typing import Optional, Any, Callable, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# Provide a local no-op @override decorator recognized by type checkers
Fn = TypeVar("Fn", bound=Callable[..., Any])
def override(func: Fn) -> Fn:  # type: ignore[misc]
    return func

# (No direct need for get_architecture/ModelConfig when loading pretrained)
import tabpfn
from tabpfn.model_loading import load_model_criterion_config


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DemoConfig:
    # Data
    n_vars: int = 8
    n_obs: int = 512
    edge_prob: float = 0.25
    noise_scale: float = 1.0
    # Model/backbone
    emsize: int = 64
    nhead: int = 8
    nlayers: int = 4
    features_per_group: int = 1  # one token per variable
    # Always load official TabPFN-v2 classifier weights; no fallback to random init
    model_path: Optional[str] = None  # keep None to auto-download/cache
    # Train
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Causal loss
    acyclicity_weight: float = 1.0
    power_iters: int = 10


# ----------------------------
# Simple synthetic SCM data (no JAX)
# ----------------------------

def sample_dag(n_vars: int, p: float, seed: Optional[int] = None) -> np.ndarray:
    """Sample a random DAG by sampling an order and edges forward with prob p."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_vars)
    A = np.zeros((n_vars, n_vars), dtype=np.float32)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if rng.random() < p:
                A[order[i], order[j]] = 1.0
    return A


def sample_linear_sem(
    A: np.ndarray, n: int, noise_scale: float = 1.0, seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Linear SEM: X = (I - W^T)^{-1} eps, weights on edges in A sampled randomly."""
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    # Sample weights for edges
    W = A * rng.uniform(low=0.3, high=1.0, size=A.shape) * rng.choice([-1.0, 1.0], size=A.shape)
    I = np.eye(d, dtype=np.float32)
    # Ensure (I - W^T) invertible (DAG ensures triangular under topological order)
    M = I - W.T
    # Solve M X^T = eps^T for X^T per sample
    eps = rng.normal(0.0, noise_scale, size=(n, d)).astype(np.float32)
    X = eps @ np.linalg.inv(M).T
    return X.astype(np.float32), W.astype(np.float32)


# ----------------------------
# AVICI-style cosine decoder (PyTorch)
# ----------------------------

class CausalGraphDecoder(nn.Module):
    """AVICI-like decoder: cosine bilinear with learned temperature and bias.

    Input:  node_emb [B, D, E]
    Output: logits [B, D, D], probs via sigmoid(logits)
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj_u = nn.Linear(emb_dim, emb_dim)
        self.proj_v = nn.Linear(emb_dim, emb_dim)
        self.ln_u = nn.LayerNorm(emb_dim)
        self.ln_v = nn.LayerNorm(emb_dim)
        # Initialize close to AVICI defaults
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    @override
    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # node_emb: [B, D, E]
        u = self.ln_u(self.proj_u(node_emb))
        v = self.ln_v(self.proj_v(node_emb))
        # L2-normalize along E
        u = F.normalize(u, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)
        # Bilinear via cosine sim: [B,D,E] x [B,E,D] -> [B,D,D]
        logits = torch.matmul(u, v.transpose(-2, -1))
        logits = logits * torch.exp(self.log_temp) + self.bias
        # Mask diagonal (no self-loops). Use a large negative so sigmoid ~ 0
        diag_mask = torch.eye(logits.size(-1), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diag_mask.unsqueeze(0), -1e4)
        return logits


# ----------------------------
# Acyclicity penalty (No BEARS-inspired, PyTorch)
# ----------------------------

def spectral_radius_power_iteration(mat: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Estimate largest eigenvalue (spectral radius) of a non-negative matrix.

    mat: [B, D, D] non-negative (we'll feed probabilities)
    Returns: [B] largest eigenvalue estimate per batch
    """
    B, D, _ = mat.shape
    # Start with random vectors
    u = torch.rand(B, D, device=mat.device, dtype=mat.dtype)
    v = torch.rand(B, D, device=mat.device, dtype=mat.dtype)
    for _ in range(iters):
        # u <- u @ mat (right eigenvector), v <- mat @ v (left eigenvector)
        u = torch.matmul(u, mat)  # [B, D]
        v = torch.matmul(mat, v.unsqueeze(-1)).squeeze(-1)  # [B, D]
        # Normalize
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-12)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
    # Rayleigh quotient approximation: (u @ (mat @ v)) / (u @ v)
    num = torch.sum(u * torch.matmul(mat, v.unsqueeze(-1)).squeeze(-1), dim=-1)
    den = torch.sum(u * v, dim=-1).clamp_min(1e-12)
    return num / den


def acyclicity_penalty_from_logits(logits: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Compute acyclicity penalty ala AVICI on edge logits.

    1) Convert to probabilities with sigmoid
    2) Zero the diagonal
    3) Spectral radius of P (No BEARS works in log-space; this is a close surrogate)

    logits: [B, D, D]
    returns: [B] penalty (>=0)
    """
    P = torch.sigmoid(logits)
    _, D, _ = P.shape
    diag_mask = torch.eye(D, device=P.device, dtype=torch.bool)
    P = P.masked_fill(diag_mask.unsqueeze(0), 0.0)
    rho = spectral_radius_power_iteration(P, iters)
    return rho


# ----------------------------
# Helper: Build a tiny TabPFN backbone
# ----------------------------

def build_tabpfn_backbone(cfg: DemoConfig) -> nn.Module:
    """Return a TabPFN backbone loaded from official pretrained weights.

    This will attempt to download/use cached TabPFN-v2 classifier weights.
    If loading fails, the error is propagated (no fallback to random init).
    """
    model, _criterion, _config = load_model_criterion_config(
        cfg.model_path,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=True,
        which="classifier",
        version="v2",
        download=True,
    )
    return model


# ----------------------------
# Extract per-variable embeddings from TabPFN
# (re-implements a minimal subset of PerFeatureTransformer.forward)
# ----------------------------

def get_node_embeddings_from_tabpfn(model: Any, X_np: np.ndarray) -> torch.Tensor:
    """Compute per-variable embeddings by running TabPFN encoders + transformer.

    X_np: [S, D] numpy (single dataset). We'll add batch dim B=1.
    Returns: node_emb [B=1, D_eff, E], where D_eff accounts for features_per_group padding
    """
    assert X_np.ndim == 2
    S, D = X_np.shape
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Build x dict like the architecture expects: x["main"]: [S,B,D]
    x_main = torch.as_tensor(X_np, device=device, dtype=dtype).unsqueeze(1)  # [S,1,D]

    # y: dummy zeros for encoder; we'll treat all rows as train for caching
    y_main = torch.zeros(S, 1, 1, device=device, dtype=dtype)  # [S,1,1]

    # Padding to multiple of features_per_group
    fpg = int(model.features_per_group)  # type: ignore
    missing = (fpg - (D % fpg)) % fpg
    if missing > 0:
        pad = torch.zeros(S, 1, missing, device=device, dtype=dtype)
        x_main = torch.cat([x_main, pad], dim=-1)
        D = D + missing

    # Rearrange to groups: x: [B,S,F,n]
    x_b_s_f_n = x_main.permute(1, 0, 2).contiguous()  # [1,S,D]
    F = D // fpg
    x_b_s_f_n = x_b_s_f_n.view(1, S, F, int(fpg))

    # Encode X: SequentialEncoder expects dict and flattened (s, b*f, n)
    x_dict = {"main": x_b_s_f_n}
    import einops  # local import
    x_flat = {k: einops.rearrange(v, "b s f n -> s (b f) n") for k, v in x_dict.items()}
    embedded_x = model.encoder(
        x_flat,
        single_eval_pos=S,
        cache_trainset_representation=True,
    )  # [s, b*f, e]
    embedded_x = einops.rearrange(embedded_x, "s (b f) e -> b s f e", b=1)  # [1,S,F,E]

    # Encode y (dummy zeros but goes through y-encoder for shape semantics)
    y_dict = {"main": y_main}
    embedded_y = model.y_encoder(
        y_dict,
        single_eval_pos=S,
        cache_trainset_representation=True,
    ).transpose(0, 1)  # [1,S,E]

    # Add embeddings (positional, DAG if configured)
    embedded_x, embedded_y = model.add_embeddings(
        embedded_x,
        embedded_y,
        data_dags=None,
        num_features=D,
        seq_len=S,
        cache_embeddings=True,
        use_cached_embeddings=False,
    )

    # Concatenate feature tokens with target token
    embedded_input = torch.cat([embedded_x, embedded_y.unsqueeze(2)], dim=2)  # [1,S,F+1,E]

    # Run encoder (treat all rows as train so single_eval_pos=S)
    enc_out = model.transformer_encoder(
        embedded_input,
        single_eval_pos=S,
        cache_trainset_representation=True,
    )  # [1,S,F+1,E]

    # Extract feature tokens and pool across S (AVICI uses max over observations)
    feature_tokens = enc_out[:, :, :-1, :]  # [1,S,F,E]
    node_emb = feature_tokens.max(dim=1).values  # [1,F,E]
    return node_emb


# ----------------------------
# Training demo
# ----------------------------

def train_demo(cfg: DemoConfig) -> None:
    set_seed(0)

    # 1) Data
    A = sample_dag(cfg.n_vars, cfg.edge_prob, seed=0)
    X_np, W = sample_linear_sem(A, cfg.n_obs, cfg.noise_scale, seed=1)
    # Ground-truth adjacency (binary)
    G = (np.abs(W) > 1e-8).astype(np.float32)

    # 2) Backbone
    model = build_tabpfn_backbone(cfg).to(cfg.device)
    model.eval()  # freeze backbone for the demo
    for p in model.parameters():
        p.requires_grad_(False)

    # 3) Decoder — match the backbone's actual embedding size
    emb_dim = int(getattr(model, "ninp", cfg.emsize))
    decoder = CausalGraphDecoder(emb_dim).to(cfg.device)
    opt = optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 4) Prepare tensors
    X_np = X_np.astype(np.float32)
    G_t = torch.as_tensor(G, device=cfg.device, dtype=torch.float32).unsqueeze(0)  # [1,D,D]

    y_np = np.zeros((cfg.n_obs, 1), dtype=np.float32)  # dummy y for backbone
    clf = tabpfn.TabPFNClassifier()  # dummy classifier for compatibility
    clf.fit(X_np, y_np)
    clf.predict(X_np)

    # 5) Train loop (decoder only)
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        # node embeddings from TabPFN backbone
        node_emb = get_node_embeddings_from_tabpfn(model, X_np)  # [1,F,E]
        logits = decoder(node_emb)  # [1,F,F]
        # BCE with logits on off-diagonals
        diag_mask = torch.eye(logits.size(-1), device=logits.device, dtype=torch.bool)
        bce = F.binary_cross_entropy_with_logits(logits[~diag_mask.unsqueeze(0)], G_t[~diag_mask.unsqueeze(0)])
        # Acyclicity penalty
        acyc = acyclicity_penalty_from_logits(logits, iters=cfg.power_iters).mean()
        loss = bce + cfg.acyclicity_weight * acyc
        loss.backward()
        opt.step()

        if (step + 1) % max(1, cfg.steps // 10) == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)[0]
                # Simple progress metric: mean absolute error of probabilities at GT edges vs non-edges
                pos_mae = (probs[G_t[0] > 0] - 1.0).abs().mean().item() if (G_t[0] > 0).any() else 0.0
                neg_mae = (probs[G_t[0] == 0] - 0.0).abs().mean().item()
            print(f"Step {step+1:04d} | loss={loss.item():.4f} | bce={bce.item():.4f} | acyc={acyc.item():.4f} | pos_mae={pos_mae:.3f} | neg_mae={neg_mae:.3f}")

    # 6) Final report
    with torch.no_grad():
        node_emb = get_node_embeddings_from_tabpfn(model, X_np)
        logits = decoder(node_emb)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        np.set_printoptions(precision=2, suppress=True)
        print("\nGround-truth adjacency (binary):")
        print(G)
        print("\nPredicted adjacency probabilities:")
        print(probs)


if __name__ == "__main__":
    cfg = DemoConfig()
    train_demo(cfg)
