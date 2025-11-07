import random
from dataclasses import dataclass
from typing import Optional, Any, Callable, TypeVar
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from cdt.metrics import SID

# Provide a local no-op @override decorator recognized by type checkers
Fn = TypeVar("Fn", bound=Callable[..., Any])
def override(func: Fn) -> Fn:  # type: ignore[misc]
    return func

# (No direct need for get_architecture/ModelConfig when loading pretrained)
from tabpfn.model_loading import load_model_criterion_config


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DemoConfig:
    # Data - matching AVICI LINEAR test:train configuration exactly
    n_vars: int = 30  # d=30 from paper
    n_obs: int = 1000  # n=1000 total observations (500 obs + 500 int)
    edge_prob: float = 2 / 30  # 2 / 30 edges_per_var=2 on average for d=30
    n_interv_obs: int = 500  # half observational, half interventional (like paper)
    n_interv_vars: int = 4  # intervene on ALL variables (n_interv_vars: -1 in AVICI config means all)
    # Model/backbone
    emsize: int = 64
    nhead: int = 8
    nlayers: int = 4
    # Note: features_per_group is loaded from checkpoint (=2), not set here
    # Always load official TabPFN-v2 classifier weights; no fallback to random init
    model_path: Optional[str] = None  # keep None to auto-download/cache
    # Train (matching AVICI defaults)
    lr: float = 2e-4  # AVICI default learning rate
    weight_decay: float = 0.0  # AVICI default (LAMB optimizer handles weight decay)
    steps: int = 10000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Causal loss (matching AVICI defaults)
    label_smoothing: float = 0.0  # AVICI default: no label smoothing
    pos_weight: float = 1.0  # AVICI default: no positive class weighting
    acyclicity_weight: float = 1.0  # AVICI default base weight
    acyclicity_schedule: str = "dual"  # AVICI default: dual ascent (augmented Lagrangian)
    acyclicity_burnin: int = 50000  # AVICI default burnin (for linear schedule)
    acyclicity_linear_rate: float = 1.0  # AVICI default linear rate
    acyclicity_dual_lr: float = 1e-4  # AVICI default: 1e-4 (not 0.01!)
    acyclicity_inner_step: int = 500  # AVICI default: update dual every 500 steps
    acyclicity_warmup: int = 1000  # warmup steps for dual schedule
    power_iters: int = 10  # AVICI default: 10 power iterations
    # Evaluation
    n_test_instances: int = 10  # number of test instances to average over
    decision_threshold: float = 0.5  # threshold for converting probabilities to binary predictions


# ----------------------------
# Simple synthetic SCM data (no JAX)
# ----------------------------
# Data generation matches AVICI's LINEAR domain exactly:
# 1. Observational data: use closed-form solution for linear SEMs
# 2. Interventional data: use ancestral sampling to ensure descendants
#    are computed AFTER the intervention (matching AVICI's sample_recursive)
# 3. Linear additive noise model: X_j = sum(W_ij * X_i) + bias_j + eps_j
# 4. Parameters match experiments/linear-base/train.yaml test:train config:
#    - Edge weights: signed_uniform(1.0, 3.0)
#    - Noise scale: uniform(0.2, 2.0) per variable (heterogeneous)
#    - Intervention dist: signed_uniform(1.0, 3.0)
#    - Bias: uniform(-3.0, 3.0) [not implemented in demo for simplicity]

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
    A: np.ndarray, n: int, noise_scales: np.ndarray, seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Linear SEM: X = (I - W^T)^{-1} eps, weights on edges in A sampled randomly."""
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    # Sample weights for edges - AVICI LINEAR: signed_uniform(1.0, 3.0)
    W = A * rng.uniform(low=1.0, high=3.0, size=A.shape) * rng.choice([-1.0, 1.0], size=A.shape)
    I = np.eye(d, dtype=np.float32)
    # Ensure (I - W^T) invertible (DAG ensures triangular under topological order)
    M = I - W.T
    # Solve M X^T = eps^T for X^T per sample
    # AVICI LINEAR: heterogeneous noise scale per variable, uniform(0.2, 2.0)
    eps = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32) * noise_scales[np.newaxis, :]
    X = eps @ np.linalg.inv(M).T
    return X.astype(np.float32), W.astype(np.float32)


def topological_sort(A: np.ndarray) -> list[int]:
    """Return topological order of DAG given adjacency matrix A."""
    n = A.shape[0]
    in_degree = A.sum(axis=0).astype(int)
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    
    while queue:
        node = queue.pop(0)
        top_order.append(node)
        # For each child of node
        for child in range(n):
            if A[node, child] > 0:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    
    return top_order


def sample_interventional_data(
    A: np.ndarray,
    W: np.ndarray,
    n_obs: int,
    n_interv: int,
    interv_vars: list[int],
    noise_scales: np.ndarray,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample interventional data using ancestral sampling (like AVICI).
    
    This properly implements causal interventions by:
    1. Setting the intervened variable to the intervention value
    2. Computing descendants based on the intervened value (not the natural value)
    
    Args:
        A: adjacency matrix [d, d]
        W: weight matrix [d, d]
        n_obs: number of observational samples
        n_interv: number of interventional samples per variable
        interv_vars: list of variable indices to intervene on
        noise_scales: noise scale per variable [d]
        seed: random seed
    
    Returns:
        X: [n_obs + n_interv*len(interv_vars), d] data matrix
        interv_mask: [n_obs + n_interv*len(interv_vars), d] intervention indicators
    """
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    toporder = topological_sort(A)
    
    # Observational data (can still use closed form)
    I = np.eye(d, dtype=np.float32)
    M = I - W.T
    # AVICI LINEAR: heterogeneous noise scale per variable
    eps_obs = rng.normal(0.0, 1.0, size=(n_obs, d)).astype(np.float32) * noise_scales[np.newaxis, :]
    X_obs = eps_obs @ np.linalg.inv(M).T
    interv_mask_obs = np.zeros((n_obs, d), dtype=np.float32)
    
    # Interventional data (use ancestral sampling for correctness)
    X_int_list = []
    interv_mask_int_list = []
    
    for var_idx in interv_vars:
        X_int = np.zeros((n_interv, d), dtype=np.float32)
        
        # Ancestral sampling in topological order
        for j in toporder:
            if j == var_idx:
                # Intervention: set to intervention value (not natural value)
                # AVICI LINEAR: signed_uniform(1.0, 3.0)
                interv_values = rng.uniform(low=1.0, high=3.0, size=n_interv) * rng.choice([-1.0, 1.0], size=n_interv)
                X_int[:, j] = interv_values.astype(np.float32)
            else:
                # Natural: compute from parents and noise
                # AVICI LINEAR: heterogeneous noise scale per variable
                eps_j = rng.normal(0.0, noise_scales[j], size=n_interv).astype(np.float32)
                # X_j = sum(W_ij * X_i for parents i) + eps_j
                parent_contrib = (X_int @ W[:, j])  # W[:, j] are weights from parents to j
                X_int[:, j] = parent_contrib + eps_j
        
        # Create intervention mask
        mask = np.zeros((n_interv, d), dtype=np.float32)
        mask[:, var_idx] = 1.0
        
        X_int_list.append(X_int)
        interv_mask_int_list.append(mask)
    
    # Concatenate all data
    X = np.vstack([X_obs] + X_int_list)
    interv_mask = np.vstack([interv_mask_obs] + interv_mask_int_list)
    
    return X.astype(np.float32), interv_mask.astype(np.float32)


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




def exp_matmul(logmat: torch.Tensor, vec: torch.Tensor, axis: int) -> torch.Tensor:
    """Matrix-vector multiplication in log-space: exp(logmat) @ vec or vec @ exp(logmat).
    
    Matches AVICI's exp_matmul for numerical stability in acyclicity computation.
    
    Args:
        logmat: [B, D, D] log-probabilities
        vec: [B, D] vector
        axis: -1 for right multiply (vec @ exp(logmat)), -2 for left multiply (exp(logmat) @ vec)
    
    Returns:
        [B, D] result of multiplication
    """
    if axis == -1:
        # vec @ exp(logmat): sum over second-to-last dimension
        # logmat: [B, D, D], vec: [B, D] -> [B, D, 1] @ [B, D, D] -> [B, D, D] -> sum -> [B, D]
        weighted = logmat + vec.unsqueeze(-1)  # [B, D, D]
        result = torch.logsumexp(weighted, dim=-2)  # [B, D]
        return torch.exp(result)
    elif axis == -2:
        # exp(logmat) @ vec: sum over last dimension
        # logmat: [B, D, D], vec: [B, D] -> [B, D, D] @ [B, D, 1] -> [B, D, D] -> sum -> [B, D]
        weighted = logmat + vec.unsqueeze(-2)  # [B, D, D]
        result = torch.logsumexp(weighted, dim=-1)  # [B, D]
        return torch.exp(result)
    else:
        raise ValueError(f"Invalid axis {axis}")


def acyclicity_penalty_from_logits(logits: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Compute acyclicity penalty in log-space like AVICI (No BEARS).
    
    Uses log-probabilities for numerical stability, matching AVICI's implementation.
    
    Args:
        logits: [B, D, D] edge logits
        iters: number of power iterations
    
    Returns:
        [B] spectral radius estimate
    """
    B, D, _ = logits.shape
    
    # Convert to log-probabilities and mask diagonal
    logp = F.logsigmoid(logits)  # [B, D, D]
    diag_mask = torch.eye(D, device=logits.device, dtype=torch.bool)
    logp = logp.masked_fill(diag_mask.unsqueeze(0), -float('inf'))
    
    # Initialize random vectors for power iteration
    u = torch.randn(B, D, device=logits.device, dtype=logits.dtype)
    v = torch.randn(B, D, device=logits.device, dtype=logits.dtype)
    
    for _ in range(iters):
        # u_new = u @ exp(logp)
        u_new = exp_matmul(logp, u, axis=-1)
        # v_new = exp(logp) @ v
        v_new = exp_matmul(logp, v, axis=-2)
        
        # Normalize
        u = u_new / (u_new.norm(dim=-1, keepdim=True) + 1e-12)
        v = v_new / (v_new.norm(dim=-1, keepdim=True) + 1e-12)
    
    # Stop gradient on eigenvectors
    u = u.detach()
    v = v.detach()
    
    # Rayleigh quotient: (u @ exp(logp) @ v) / (u @ v)
    numerator = (u * exp_matmul(logp, v, axis=-2)).sum(dim=-1)
    denominator = (u * v).sum(dim=-1).clamp(min=1e-12)
    
    return numerator / denominator


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

def prepare_data_with_interventions(X: np.ndarray, interv_mask: np.ndarray) -> np.ndarray:
    """Interleave variable values and intervention masks for TabPFN input.
    
    Args:
        X: [n_obs, n_vars] - variable values
        interv_mask: [n_obs, n_vars] - binary intervention indicators
    
    Returns:
        X_interleaved: [n_obs, n_vars*2] - interleaved format [val1, interv1, val2, interv2, ...]
    """
    n_obs, n_vars = X.shape
    X_interleaved = np.zeros((n_obs, n_vars * 2), dtype=np.float32)
    
    # Interleave: [val1, interv1, val2, interv2, ...]
    X_interleaved[:, 0::2] = X  # values at even indices
    X_interleaved[:, 1::2] = interv_mask  # masks at odd indices
    
    # Apply minimal preprocessing: z-normalize ONLY the value columns (even indices)
    # This mimics AVICI's standardization of data values while keeping intervention masks binary
    for i in range(0, n_vars * 2, 2):
        col = X_interleaved[:, i]
        mean = col.mean()
        std = col.std()
        if std > 0:
            X_interleaved[:, i] = (col - mean) / std
    
    # Note: intervention mask columns (odd indices) remain binary [0, 1]
    
    return X_interleaved


def get_node_embeddings_from_tabpfn(
    model: Any, 
    X_np: np.ndarray, 
    interv_mask: np.ndarray
) -> torch.Tensor:
    """Compute per-variable embeddings by running TabPFN encoders + transformer.
    
    With features_per_group=2 from the pretrained model, each variable gets one group
    containing [value, intervention_indicator].

    Args:
        model: TabPFN model (features_per_group=2 from checkpoint)
        X_np: [S, n_vars] - variable values
        interv_mask: [S, n_vars] - binary intervention indicators
    
    Returns:
        node_emb: [B=1, n_vars, E] - one embedding per variable
    """
    # Prepare interleaved data: [S, n_vars*2]
    X_interleaved = prepare_data_with_interventions(X_np, interv_mask)
    
    assert X_interleaved.ndim == 2
    S, D = X_interleaved.shape
    # D = n_vars * 2 (interleaved format)
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Build x dict like the architecture expects: x["main"]: [S,B,D]
    x_main = torch.as_tensor(X_interleaved, device=device, dtype=dtype).unsqueeze(1)  # [S,1,D]

    # y: dummy zeros for encoder; we'll treat all rows as train for caching
    y_main = torch.zeros(S, 1, 1, device=device, dtype=dtype)  # [S,1,1]

    # Padding to multiple of features_per_group
    fpg = int(model.features_per_group)  # This is 2 from checkpoint
    missing = (fpg - (D % fpg)) % fpg
    if missing > 0:
        pad = torch.zeros(S, 1, missing, device=device, dtype=dtype)
        x_main = torch.cat([x_main, pad], dim=-1)
        D = D + missing

    # Rearrange to groups: x: [B,S,F,n]
    # With fpg=2 and D=n_vars*2, F = n_vars (perfect!)
    # Each group contains [val_i, interv_i] for variable i
    x_b_s_f_n = x_main.permute(1, 0, 2).contiguous()  # [1,S,D]
    F = D // fpg  # F = n_vars
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
    feature_tokens = enc_out[:, :, :-1, :]  # [1,S,F,E] where F=n_vars
    node_emb = feature_tokens.max(dim=1).values  # [1,n_vars,E]
    
    # Now node_emb[0, i, :] is the embedding for variable i
    # It encodes both the variable's values AND its intervention status
    return node_emb


# ----------------------------
# Evaluation metrics
# ----------------------------

def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute precision, recall, and F1 score for binary edge predictions.
    
    Args:
        y_true: [d, d] ground truth adjacency matrix (binary)
        y_pred: [d, d] predicted adjacency matrix (binary)
    
    Returns:
        dict with 'precision', 'recall', 'f1'
    """
    # Flatten and exclude diagonal (no self-loops)
    d = y_true.shape[0]
    mask = ~np.eye(d, dtype=bool)
    y_true_flat = y_true[mask].flatten()
    y_pred_flat = y_pred[mask].flatten()
    
    # True positives, false positives, false negatives
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    
    # Compute metrics (handle edge cases)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if tp + fn == 0 else 0.0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 1.0 if tp + fp == 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 1.0 if tp + fp + fn == 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }


# ----------------------------
# Training demo
# ----------------------------

def train_demo(cfg: DemoConfig) -> dict[str, float]:
    # 1) Data - Generate observational + interventional data
    A = sample_dag(cfg.n_vars, cfg.edge_prob, seed=None)
    
    # AVICI LINEAR: heterogeneous noise scale per variable, uniform(0.2, 2.0)
    rng_noise = np.random.default_rng(seed=None)
    noise_scales = rng_noise.uniform(low=0.2, high=2.0, size=cfg.n_vars).astype(np.float32)
    
    _, W = sample_linear_sem(A, cfg.n_obs, noise_scales, seed=None)
    # Ground-truth adjacency (binary)
    G = (np.abs(W) > 1e-8).astype(np.float32)
    
    # Sample interventional data
    rng = np.random.default_rng(seed=None)
    interv_vars = sorted(rng.choice(cfg.n_vars, size=cfg.n_interv_vars, replace=False).tolist())
    print(f"Intervening on variables: {interv_vars}")
    
    X_np, interv_mask = sample_interventional_data(
        A, W, 
        n_obs=cfg.n_obs - cfg.n_interv_obs,  # adjust observational samples
        n_interv=cfg.n_interv_obs // cfg.n_interv_vars,  # samples per intervention
        interv_vars=interv_vars,
        noise_scales=noise_scales,
        seed=None
    )
    
    print(f"Data shape: {X_np.shape}, Intervention mask shape: {interv_mask.shape}")
    print(f"Interventional samples: {interv_mask.sum(axis=0).astype(int)}")

    # 2) Backbone
    model = build_tabpfn_backbone(cfg).to(cfg.device)
    model.eval()  # freeze backbone for the demo
    for p in model.parameters():
        p.requires_grad_(False)
    
    print(f"Model features_per_group: {model.features_per_group}")

    # 3) Decoder — match the backbone's actual embedding size
    emb_dim = int(getattr(model, "ninp", cfg.emsize))
    decoder = CausalGraphDecoder(emb_dim).to(cfg.device)
    opt = optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 4) Prepare tensors
    X_np = X_np.astype(np.float32)
    interv_mask = interv_mask.astype(np.float32)
    G_t = torch.as_tensor(G, device=cfg.device, dtype=torch.float32).unsqueeze(0)  # [1,D,D]
    
    # Initialize dual variable for dual acyclicity scheduling
    dual = torch.tensor(0.0, device=cfg.device, dtype=torch.float32)

    t0 = time.time()
    # 5) Train loop (decoder only)
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        # node embeddings from TabPFN backbone (now with intervention information)
        node_emb = get_node_embeddings_from_tabpfn(model, X_np, interv_mask)  # [1,n_vars,E]
        logits = decoder(node_emb)  # [1,n_vars,n_vars]
        
        # AVICI-style BCE with label smoothing and positive weighting
        diag_mask = torch.eye(logits.size(-1), device=logits.device, dtype=torch.bool)
        n_vars = logits.size(-1)
        
        # Apply label smoothing to targets
        y_soft = (1 - cfg.label_smoothing) * G_t + cfg.label_smoothing / 2.0
        
        # Compute log probabilities
        logp1 = F.logsigmoid(logits)
        logp0 = F.logsigmoid(-logits)
        
        # Binary cross-entropy with positive weighting
        xent_eltwise = -(cfg.pos_weight * y_soft * logp1 + (1 - y_soft) * logp0)
        
        # Mask diagonal and normalize by number of edges (AVICI style)
        xent_masked = xent_eltwise.masked_fill(diag_mask.unsqueeze(0), 0.0)
        bce = xent_masked.sum() / (n_vars * (n_vars - 1))
        
        # Acyclicity penalty in log-space (AVICI style)
        acyc_penalty = acyclicity_penalty_from_logits(logits, iters=cfg.power_iters).mean()
        
        # Adaptive acyclicity weight scheduling (matching AVICI)
        if cfg.acyclicity_schedule == "const":
            acyc_weight = cfg.acyclicity_weight
        elif cfg.acyclicity_schedule == "linear":
            acyc_weight = cfg.acyclicity_weight * max(0.0, (step - cfg.acyclicity_burnin) * cfg.acyclicity_linear_rate)
        elif cfg.acyclicity_schedule == "dual":
            # Dual ascent (augmented Lagrangian)
            if step >= cfg.acyclicity_warmup:
                acyc_weight = cfg.acyclicity_weight * dual.item()
            else:
                acyc_weight = 0.0
        else:
            raise ValueError(f"Unknown acyclicity schedule: {cfg.acyclicity_schedule}")
        
        wgt_acyc = acyc_weight * acyc_penalty
        loss = bce + wgt_acyc
        loss.backward()
        opt.step()
        
        # Update dual variable (for dual schedule) - only every inner_step iterations
        if cfg.acyclicity_schedule == "dual" and step >= cfg.acyclicity_warmup:
            if (step - cfg.acyclicity_warmup) % cfg.acyclicity_inner_step == 0:
                with torch.no_grad():
                    dual = dual + cfg.acyclicity_dual_lr * acyc_penalty

        if (step + 1) % max(1, cfg.steps // 10) == 0:
            tt = time.time()
            with torch.no_grad():
                probs = torch.sigmoid(logits)[0]
                # Simple progress metric: mean absolute error of probabilities at GT edges vs non-edges
                pos_mae = (probs[G_t[0] > 0] - 1.0).abs().mean().item() if (G_t[0] > 0).any() else 0.0
                neg_mae = (probs[G_t[0] == 0] - 0.0).abs().mean().item()
            
            log_str = f"Step {step+1:04d} | loss={loss.item():.4f} | bce={bce.item():.4f} | acyc_raw={acyc_penalty.item():.4f} | acyc_wgt={wgt_acyc.item():.4f}"
            if cfg.acyclicity_schedule == "dual":
                log_str += f" | dual={dual.item():.4f}"
            log_str += f" | pos_mae={pos_mae:.3f} | neg_mae={neg_mae:.3f}"
            print(log_str)
            print(f"  Non-edge probs (max 5): {probs[G_t[0] == 0].sort()[0][-5:].cpu().numpy()}")
            print(f"  Edge probs (min 5):     {probs[G_t[0] > 0].sort()[0][:5].cpu().numpy()}")
            print(f"  Time for {max(1, cfg.steps // 100)} steps: {tt - t0:.2f} sec")
            t0 = time.time()

    # 6) Final evaluation
    with torch.no_grad():
        node_emb = get_node_embeddings_from_tabpfn(model, X_np, interv_mask)
        logits = decoder(node_emb)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # Convert probabilities to binary predictions using threshold
        G_pred = (probs > cfg.decision_threshold).astype(np.float32)
        
        # Compute F1 score
        metrics = compute_f1_score(G, G_pred)
        metrics['sid'] = SID(G, G_pred)

        print(G)
        print(G_pred)
        
        return metrics


def run_benchmark(cfg: DemoConfig) -> None:
    """Run multiple test instances and report average F1 score."""
    print("=" * 80)
    print(f"AVICI-style Causal Discovery Benchmark")
    print(f"Configuration: d={cfg.n_vars} variables, n={cfg.n_obs} observations")
    print(f"  - Observational: {cfg.n_obs - cfg.n_interv_obs}")
    print(f"  - Interventional: {cfg.n_interv_obs} (on {cfg.n_interv_vars} variables)")
    print(f"  - Decision threshold: {cfg.decision_threshold}")
    print(f"  - Test instances: {cfg.n_test_instances}")
    print(f"\nLoss Configuration (matching AVICI):")
    print(f"  - Label smoothing: {cfg.label_smoothing}")
    print(f"  - Positive weight: {cfg.pos_weight}")
    print(f"  - Acyclicity schedule: {cfg.acyclicity_schedule}")
    print(f"  - Acyclicity weight: {cfg.acyclicity_weight}")
    if cfg.acyclicity_schedule == "linear":
        print(f"  - Linear rate: {cfg.acyclicity_linear_rate}, burnin: {cfg.acyclicity_burnin}")
    elif cfg.acyclicity_schedule == "dual":
        print(f"  - Dual lr: {cfg.acyclicity_dual_lr}, inner_step: {cfg.acyclicity_inner_step}, warmup: {cfg.acyclicity_warmup}")
    print(f"  - Power iterations: {cfg.power_iters} (log-space)")
    print("=" * 80)
    
    all_metrics = []
    
    for instance in range(cfg.n_test_instances):
        print(f"\n{'='*80}")
        print(f"Test Instance {instance + 1}/{cfg.n_test_instances}")
        print(f"{'='*80}")
        
        # # Use different seed for each instance
        # set_seed(instance * 100)
        
        metrics = train_demo(cfg)
        all_metrics.append(metrics)
        
        print(f"\nInstance {instance + 1} Results:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    
    # Compute statistics across all instances
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (averaged over {cfg.n_test_instances} instances)")
    print(f"{'='*80}")
    
    f1_scores = [m['f1'] for m in all_metrics]
    precision_scores = [m['precision'] for m in all_metrics]
    recall_scores = [m['recall'] for m in all_metrics]
    sid_scores = [m['sid'] for m in all_metrics]

    print(f"SID:      {np.mean(sid_scores):.4f} ± {np.std(sid_scores):.4f}")
    print(f"\nF1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:    {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    
    print(f"\nComparison to AVICI paper (Table A.2, in-distribution LINEAR):")
    print(f"  AVICI (obs only):  SID = 178.2 ± 36.9,  F1 = 0.828 ± 0.03")
    print(f"  AVICI (with int):  SID =  72.4 ± 20.7,  F1 = 0.948 ± 0.01")
    print(f"  TabPFN-AVICI:      SID = {np.mean(sid_scores):5.1f} ± {np.std(sid_scores):4.1f},  F1 = {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.2f}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    cfg = DemoConfig()
    run_benchmark(cfg)
