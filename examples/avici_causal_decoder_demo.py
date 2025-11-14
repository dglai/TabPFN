import random
from dataclasses import dataclass, asdict
from typing import Optional, Any, Callable, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from gadjid import sid as compute_sid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Provide a local no-op @override decorator recognized by type checkers
Fn = TypeVar("Fn", bound=Callable[..., Any])
def override(func: Fn) -> Fn:  # type: ignore[misc]
    return func

# (No direct need for get_architecture/ModelConfig when loading pretrained)
from tabpfn.model_loading import load_model_criterion_config

# Weights & Biases logging (assumed installed and always enabled)
import wandb


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DemoConfig:
    # Data - matching AVICI LINEAR test:train configuration exactly
    n_vars: int = 6  # d=30 from paper
    n_obs: int = 16  # n=1000 total observations (500 obs + 500 int)
    edge_prob: float = 2 / 6  # 2 / 30 edges_per_var=2 on average for d=30
    n_interv_obs: int = 8  # half observational, half interventional (like paper)
    n_interv_vars: int = 2  # intervene on ALL variables (n_interv_vars: -1 in AVICI config means all)
    # Evaluation
    n_test_instances: int = 1  # number of test instances to average over
    decision_threshold: float = 0.5  # threshold for converting probabilities to binary predictions
    # Always load official TabPFN-v2 classifier weights; no fallback to random init
    model_path: Optional[str] = None  # keep None to auto-download/cache

    # Checkpointing
    checkpoint_dir: str = "~/autodl-tmp/checkpoints"  # directory to save checkpoints
    checkpoint_interval: int = 1000  # checkpoint every N steps

    # Logging (evaluation/monitoring)
    wandb_enabled: bool = True
    wandb_project: str = "tabpfn-avici-demo"
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_notes: Optional[str] = None


@dataclass
class Hyperparameters:
    # Model/backbone
    emsize: int = 64
    nhead: int = 8
    nlayers: int = 4
    # Train (matching AVICI defaults)
    lr: float = 1e-4  # AVICI LINEAR base learning rate
    weight_decay: float = 0.0  # AVICI default (LAMB optimizer handles weight decay)
    steps: int = 1000000  # Increased to 100k (AVICI uses 300k, but 100k for faster testing)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 48  # Number of graphs per batch
    # DataLoader configuration (cache-on-the-fly mode)
    num_batches: Optional[int] = 10  # Max batches to cache (None = infinite, never caches)
    # Evaluation configuration
    eval_num_batches: int = 1  # Number of batches for evaluation set
    # Optimizer (matching AVICI defaults)
    grad_clip: float = 1.0  # AVICI default: clip gradients at 1.0
    lr_schedule: str = "piecewise"  # AVICI default: piecewise_const_200k_300k
    lr_drop_steps: Optional[list[int]] = None  # Will be set to [70000] for 100k training
    lr_drop_factor: float = 0.1  # AVICI default: multiply by 0.1 at each drop
    polyak_step_size: float = 0.001  # AVICI default: EMA rate for parameter averaging
    # Causal loss (matching AVICI defaults)
    label_smoothing: float = 0.0  # AVICI default: no label smoothing
    pos_weight: float = 1.0  # AVICI default: no positive class weighting
    acyclicity_weight: float = 1.0  # AVICI default base weight
    acyclicity_schedule: str = "dual"  # AVICI default: dual ascent (augmented Lagrangian)
    acyclicity_burnin: int = 50000  # AVICI default burnin
    acyclicity_linear_rate: float = 1.0  # AVICI default linear rate
    acyclicity_dual_lr: float = 1e-4  # AVICI default: 1e-4
    acyclicity_inner_step: int = 500  # AVICI default: update dual every 500 steps
    acyclicity_warmup: bool = True  # AVICI default: warmup dual_lr gradually
    acyclicity_polyak: float = 1e-4  # AVICI default: Polyak averaging rate for penalty
    power_iters: int = 10  # AVICI default: 10 power iterations
    
    def __post_init__(self):
        # Set default LR drop steps if not provided
        if self.lr_drop_steps is None:
            # For 100k training, drop at 70k (AVICI drops at 200k, 300k for 300k training)
            self.lr_drop_steps = [70000]


# ----------------------------
# Causal Graph Dataset (PyTorch)
# ----------------------------

class CausalGraphDataset(Dataset):
    """PyTorch Dataset for generating causal graphs with interventional data.
    
    Supports two modes:
    1. Fixed size: Pre-generates a fixed number of graphs
    2. Infinite: Generates graphs on-the-fly (for online training)
    """
    
    def __init__(
        self,
        cfg: DemoConfig,
        num_graphs: Optional[int] = None,
    ):
        super().__init__()
        """Initialize the Dataset.
        
        Args:
            cfg: Demo configuration
            num_graphs: Maximum number of graphs to cache. If None, generates infinitely without caching.
        """
        self.cfg = cfg
        self.num_graphs = num_graphs
        
        # Start with empty cache (will fill on-the-fly if num_graphs is set)
        self.graphs: list[dict[str, np.ndarray]] = []
        self.is_cache_full = False
        self.cache_indices: list[int] = []  # Shuffled indices for cache access
        self.current_cache_pos = 0  # Current position in shuffled cache
    
    def _generate_single_graph(self) -> dict[str, np.ndarray]:
        """Generate a single graph with data.
        
        Returns:
            Dictionary with keys: 'A', 'G', 'X', 'interv_mask'
        """
        A = sample_dag(self.cfg.n_vars, self.cfg.edge_prob, seed=None)
        
        # AVICI LINEAR: heterogeneous noise scale per variable
        rng_noise = np.random.default_rng(seed=None)
        noise_scales = rng_noise.uniform(low=0.2, high=2.0, size=self.cfg.n_vars).astype(np.float32)
        
        _, W = sample_linear_sem(A, self.cfg.n_obs, noise_scales, seed=None)
        G = (np.abs(W) > 1e-8).astype(np.float32)
        
        # Sample interventional data
        rng = np.random.default_rng(seed=None)
        interv_vars = sorted(rng.choice(self.cfg.n_vars, size=self.cfg.n_interv_vars, replace=False).tolist())
        
        X_np, interv_mask = sample_interventional_data(
            A, W,
            n_obs=self.cfg.n_obs - self.cfg.n_interv_obs,
            n_interv=self.cfg.n_interv_obs // self.cfg.n_interv_vars,
            interv_vars=interv_vars,
            noise_scales=noise_scales,
            seed=None
        )
        
        return {
            'A': A,
            'G': G,
            'X': X_np,
            'interv_mask': interv_mask,
        }
    
    @override
    def __len__(self) -> int:
        """Return dataset size."""
        if self.num_graphs is not None:
            return self.num_graphs
        else:
            # For online generation, return a large number
            return h.steps  # Effectively infinite
    
    def _shuffle_cache(self) -> None:
        """Shuffle the cache indices for random access."""
        self.cache_indices = list(range(len(self.graphs)))
        random.shuffle(self.cache_indices)
        self.current_cache_pos = 0
    
    @override
    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:  # type: ignore[override]
        """Get a single graph.
        
        Cache-on-the-fly behavior with shuffling:
        - If num_graphs is None: Always generates new graphs (infinite, no caching)
        - If num_graphs is set and cache not full: Generate and cache new graph
        - If cache is full: Return from shuffled cache, reshuffle when all graphs are used
        
        Args:
            idx: Index (used for building cache, ignored when looping over shuffled cache)
            
        Returns:
            Dictionary with keys: 'A', 'G', 'X', 'interv_mask'
        """
        if self.num_graphs is None:
            # Infinite mode: always generate new graphs, never cache
            return self._generate_single_graph()
        
        # Cache-on-the-fly mode (num_graphs is set)
        if not self.is_cache_full:
            # Still building cache
            if idx < len(self.graphs):
                # Already cached
                return self.graphs[idx]
            else:
                # Generate new graph and cache it
                graph = self._generate_single_graph()
                self.graphs.append(graph)
                
                # Check if cache is now full
                if len(self.graphs) >= self.num_graphs:
                    self.is_cache_full = True
                    print(f"✓ Cache full: {len(self.graphs)} graphs cached. Shuffling and looping over cached data.")
                    self._shuffle_cache()  # Initial shuffle
                
                return graph
        else:
            # Cache is full, use shuffled indices
            # Get current graph from shuffled cache
            cache_idx = self.cache_indices[self.current_cache_pos]
            graph = self.graphs[cache_idx]
            
            # Move to next position
            self.current_cache_pos += 1
            
            # If we've gone through all cached graphs, reshuffle
            if self.current_cache_pos >= len(self.graphs):
                self._shuffle_cache()
            
            return graph


def collate_causal_graphs(batch: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Collate function for CausalGraphDataset.
    
    Converts a list of individual graphs into a batched format.
    
    Args:
        batch: List of graph dictionaries
        
    Returns:
        Batched dictionary with:
            - 'A': list of adjacency matrices
            - 'G': stacked binary adjacency [batch_size, n_vars, n_vars]
            - 'X': list of data matrices
            - 'interv_mask': list of intervention masks
    """
    batch_A = [item['A'] for item in batch]
    batch_G = np.stack([item['G'] for item in batch], axis=0)
    batch_X = [item['X'] for item in batch]
    batch_interv_mask = [item['interv_mask'] for item in batch]
    
    return {
        'A': batch_A,
        'G': batch_G,
        'X': batch_X,
        'interv_mask': batch_interv_mask,
    }


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
    
    AVICI uses JAX's logsumexp(a, b=weights) which computes: log(sum(weights * exp(a)))
    PyTorch doesn't have the b parameter, so we implement it as: logsumexp(a + log(weights))
    
    Args:
        logmat: [B, D, D] log-probabilities
        vec: [B, D] vector (in NORMAL space, not log-space)
        axis: -1 for left multiply (exp(logmat) @ vec), -2 for right multiply (vec @ exp(logmat))
              This matches AVICI's convention!
    
    Returns:
        [B, D] result of multiplication
    """
    # Convert vec to log-space for weighted logsumexp
    log_vec = torch.log(vec.clamp(min=1e-45))  # Clamp to avoid log(0)
    
    if axis == -1:
        # exp(logmat) @ vec: sum over last dimension
        # result[i] = sum_j vec[j] * exp(logmat[i,j])
        # In log-space: log(result[i]) = logsumexp_j(log(vec[j]) + logmat[i,j])
        weighted = logmat + log_vec.unsqueeze(-2)  # [B, D, D]
        result = torch.logsumexp(weighted, dim=-1)  # [B, D]
        return torch.exp(result)
    elif axis == -2:
        # vec @ exp(logmat): sum over second-to-last dimension  
        # result[j] = sum_i vec[i] * exp(logmat[i,j])
        # In log-space: log(result[j]) = logsumexp_i(log(vec[i]) + logmat[i,j])
        weighted = logmat + log_vec.unsqueeze(-1)  # [B, D, D]
        result = torch.logsumexp(weighted, dim=-2)  # [B, D]
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
    
    # Initialize random positive vectors for power iteration
    u = torch.rand(B, D, device=logits.device, dtype=logits.dtype)
    v = torch.rand(B, D, device=logits.device, dtype=logits.dtype)
    
    for _ in range(iters):
        # u_new = u @ exp(logp)  (right multiply)
        u_new = exp_matmul(logp, u, axis=-2)
        # v_new = exp(logp) @ v  (left multiply)
        v_new = exp_matmul(logp, v, axis=-1)
        
        # Normalize
        u = u_new / (u_new.norm(dim=-1, keepdim=True) + 1e-12)
        v = v_new / (v_new.norm(dim=-1, keepdim=True) + 1e-12)
    
    # Stop gradient on eigenvectors
    u = u.detach()
    v = v.detach()
    
    # Rayleigh quotient: (u @ exp(logp) @ v) / (u @ v)
    # exp(logp) @ v is left multiply (axis=-1)
    numerator = (u * exp_matmul(logp, v, axis=-1)).sum(dim=-1)
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


def get_node_embeddings_from_tabpfn_batched(
    model: Any,
    batch_X: list[np.ndarray],
    batch_interv_mask: list[np.ndarray]
) -> torch.Tensor:
    """Compute per-variable embeddings for a batch of graphs using TabPFN's batch dimension.
    
    This is more efficient than processing graphs sequentially as it uses TabPFN's
    native batch processing capability.
    
    Args:
        model: TabPFN model (features_per_group=2 from checkpoint)
        batch_X: list of [S, n_vars] - variable values for each graph
        batch_interv_mask: list of [S, n_vars] - intervention indicators for each graph
    
    Returns:
        node_emb: [batch_size, n_vars, E] - embeddings for all graphs
    """
    import einops  # local import
    
    batch_size = len(batch_X)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Prepare interleaved data for all graphs
    batch_X_interleaved = []
    for X_np, interv_mask in zip(batch_X, batch_interv_mask):
        X_interleaved = prepare_data_with_interventions(X_np, interv_mask)
        batch_X_interleaved.append(X_interleaved)
    
    # All graphs should have same shape
    S, D_orig = batch_X_interleaved[0].shape
    # D_orig = n_vars * 2 (interleaved format)
    
    # Stack into TabPFN's batch format: [S, batch_size, D]
    x_main = torch.stack([
        torch.as_tensor(X_int, device=device, dtype=dtype) 
        for X_int in batch_X_interleaved
    ], dim=1)  # [S, batch_size, D]
    
    # y: dummy zeros for encoder
    y_main = torch.zeros(S, batch_size, 1, device=device, dtype=dtype)  # [S, batch_size, 1]
    
    # Padding to multiple of features_per_group
    fpg = int(model.features_per_group)  # This is 2 from checkpoint
    D = D_orig
    missing = (fpg - (D % fpg)) % fpg
    if missing > 0:
        pad = torch.zeros(S, batch_size, missing, device=device, dtype=dtype)
        x_main = torch.cat([x_main, pad], dim=-1)
        D = D + missing
    
    # Rearrange to groups: [B, S, F, n]
    # With fpg=2 and D=n_vars*2, F = n_vars
    x_b_s_f_n = x_main.permute(1, 0, 2).contiguous()  # [batch_size, S, D]
    F = D // fpg  # F = n_vars
    x_b_s_f_n = x_b_s_f_n.view(batch_size, S, F, int(fpg))
    
    # Encode X: SequentialEncoder expects dict and flattened (s, b*f, n)
    x_dict = {"main": x_b_s_f_n}
    x_flat = {k: einops.rearrange(v, "b s f n -> s (b f) n") for k, v in x_dict.items()}
    embedded_x = model.encoder(
        x_flat,
        single_eval_pos=S,
        cache_trainset_representation=True,
    )  # [s, batch_size*f, e]
    embedded_x = einops.rearrange(embedded_x, "s (b f) e -> b s f e", b=batch_size)  # [batch_size, S, F, E]
    
    # Encode y (dummy zeros)
    y_dict = {"main": y_main}
    embedded_y = model.y_encoder(
        y_dict,
        single_eval_pos=S,
        cache_trainset_representation=True,
    ).transpose(0, 1)  # [batch_size, S, E]
    
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
    embedded_input = torch.cat([embedded_x, embedded_y.unsqueeze(2)], dim=2)  # [batch_size, S, F+1, E]
    
    # Run transformer encoder
    enc_out = model.transformer_encoder(
        embedded_input,
        single_eval_pos=S,
        cache_trainset_representation=True,
    )  # [batch_size, S, F+1, E]
    
    # Extract feature tokens and pool across S (AVICI uses max over observations)
    feature_tokens = enc_out[:, :, :-1, :]  # [batch_size, S, F, E] where F=n_vars
    node_emb = feature_tokens.max(dim=1).values  # [batch_size, n_vars, E]
    
    # Now node_emb[b, i, :] is the embedding for variable i in graph b
    return node_emb


# ----------------------------
# Evaluation metrics
# ----------------------------

def is_dag(adjacency_matrix: np.ndarray) -> bool:
    """Check if adjacency matrix represents a DAG (no cycles).
    
    Args:
        adjacency_matrix: [d, d] binary adjacency matrix
    
    Returns:
        True if DAG, False if contains cycles
    """
    import networkx as nx
    G = nx.DiGraph(adjacency_matrix)
    return nx.is_directed_acyclic_graph(G)


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


def compute_sid_safe(y_true: np.ndarray, y_pred: np.ndarray, edge_direction: str = "from row to column") -> dict[str, float]:
    """Compute SID with graceful handling of cyclic graphs.
    
    If prediction is a DAG: computes SID normally
    If prediction has cycles: falls back to SHD and marks SID as invalid
    
    Args:
        y_true: [d, d] ground truth adjacency matrix (binary)
        y_pred: [d, d] predicted adjacency matrix (binary)
        edge_direction: edge direction convention for gadjid
    
    Returns:
        dict with 'sid', 'sid_normalised', 'shd', 'is_dag', 'has_cycles'
    """
    
    if is_dag(y_pred):
        sid_normalised, sid_count = compute_sid(y_true, y_pred, edge_direction)
        return {
            'sid': sid_count,
            'sid_normalised': sid_normalised,
            'is_dag': 1.0,
        }
    else:
        return {
            'sid': float('nan'),  # Invalid
            'sid_normalised': float('nan'),  # Invalid
            'is_dag': 0.0,
        }


# ----------------------------
# PyTorch Lightning Module
# ----------------------------

class CausalDecoderLightningModule(pl.LightningModule):
    """PyTorch Lightning module for causal graph decoder training."""
    
    def __init__(
        self,
        cfg: DemoConfig,
        h: Hyperparameters,
        backbone: nn.Module,
        emb_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.h = h
        self.backbone = backbone
        self.decoder = CausalGraphDecoder(emb_dim)
        
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        
        # Initialize Polyak-averaged parameters (EMA)
        self.register_buffer('ave_params_initialized', torch.tensor(False))
        self.ave_params: dict[str, torch.Tensor] = {}
        
        # Initialize dual variable and Polyak-averaged penalty for dual acyclicity scheduling
        self.register_buffer('dual', torch.tensor(0.0))
        self.register_buffer('dual_penalty_polyak', torch.tensor(0.0))
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['backbone'])
    
    @override
    def forward(self, batch_X: list[np.ndarray], batch_interv_mask: list[np.ndarray]) -> torch.Tensor:  # type: ignore[override]
        """Forward pass: get node embeddings and decode to logits."""
        node_emb_batch = get_node_embeddings_from_tabpfn_batched(
            self.backbone, batch_X, batch_interv_mask
        )
        logits = self.decoder(node_emb_batch)
        return logits
    
    @override
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Training step."""
        # Initialize Polyak-averaged parameters on first step
        if not self.ave_params_initialized:
            self.ave_params = {name: param.clone().detach() for name, param in self.decoder.named_parameters()}
            self.ave_params_initialized = torch.tensor(True)
        
        # Extract batch data
        batch_X = batch['X']
        batch_interv_mask = batch['interv_mask']
        G_batch = batch['G']
        batch_size = len(batch_X)
        
        # Prepare ground truth tensor
        G_t = torch.as_tensor(G_batch, device=self.device, dtype=torch.float32)
        
        # Forward pass
        logits = self(batch_X, batch_interv_mask)
        
        # AVICI-style BCE with label smoothing and positive weighting
        diag_mask = torch.eye(logits.size(-1), device=logits.device, dtype=torch.bool)
        n_vars = logits.size(-1)
        
        # Apply label smoothing to targets
        y_soft = (1 - self.h.label_smoothing) * G_t + self.h.label_smoothing / 2.0
        
        # Compute log probabilities
        logp1 = F.logsigmoid(logits)
        logp0 = F.logsigmoid(-logits)
        
        # Binary cross-entropy with positive weighting
        xent_eltwise = -(self.h.pos_weight * y_soft * logp1 + (1 - y_soft) * logp0)
        
        # Mask diagonal and normalize by number of edges (AVICI style)
        xent_masked = xent_eltwise.masked_fill(diag_mask.unsqueeze(0), 0.0)
        bce = xent_masked.sum(dim=(-2, -1)).mean() / (n_vars * (n_vars - 1))
        
        # Acyclicity penalty in log-space (AVICI style)
        acyc_penalty = acyclicity_penalty_from_logits(logits, iters=self.h.power_iters).mean()
        
        # Adaptive acyclicity weight scheduling (matching AVICI)
        step = self.global_step
        if self.h.acyclicity_schedule == "const":
            acyc_weight = self.h.acyclicity_weight
        elif self.h.acyclicity_schedule == "linear":
            acyc_weight = self.h.acyclicity_weight * max(0.0, (step - self.h.acyclicity_burnin) * self.h.acyclicity_linear_rate)
        elif self.h.acyclicity_schedule == "dual":
            acyc_weight = self.h.acyclicity_weight * self.dual.item()
        else:
            raise ValueError(f"Unknown acyclicity schedule: {self.h.acyclicity_schedule}")
        
        wgt_acyc = acyc_weight * acyc_penalty
        loss = bce + wgt_acyc
        
        # Update Polyak-averaged parameters (EMA)
        with torch.no_grad():
            for name, param in self.decoder.named_parameters():
                self.ave_params[name].mul_(1 - self.h.polyak_step_size).add_(param, alpha=self.h.polyak_step_size)
        
        # Update dual variable (for dual schedule)
        if self.h.acyclicity_schedule == "dual":
            with torch.no_grad():
                # Polyak averaging of acyclicity penalty
                if step == 0:
                    self.dual_penalty_polyak = acyc_penalty.clone()
                else:
                    self.dual_penalty_polyak = (1 - self.h.acyclicity_polyak) * self.dual_penalty_polyak + self.h.acyclicity_polyak * acyc_penalty
                
                # Dual learning rate with warmup
                if self.h.acyclicity_warmup:
                    dual_lr = min(self.h.acyclicity_dual_lr, step * self.h.acyclicity_dual_lr / self.h.acyclicity_burnin)
                    effective_burnin = 0
                else:
                    dual_lr = self.h.acyclicity_dual_lr
                    effective_burnin = self.h.acyclicity_burnin
                
                # Update dual every inner_step iterations after burnin
                if (step % self.h.acyclicity_inner_step == 0) and (step > effective_burnin):
                    self.dual = self.dual + dual_lr * self.dual_penalty_polyak
        
        # Logging
        with torch.no_grad():
            probs = torch.sigmoid(logits)[0]
            pos_mae = (probs[G_t[0] > 0] - 1.0).abs().mean().item() if (G_t[0] > 0).any() else 0.0
            neg_mae = (probs[G_t[0] == 0] - 0.0).abs().mean().item()
        
        self.log('loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('bce', bce, batch_size=batch_size)
        self.log('acyclicity/raw', acyc_penalty, batch_size=batch_size)
        self.log('acyclicity/weight', acyc_weight, batch_size=batch_size)
        self.log('acyclicity/weighted', wgt_acyc, batch_size=batch_size)
        self.log('metrics/pos_mae', pos_mae, batch_size=batch_size)
        self.log('metrics/neg_mae', neg_mae, batch_size=batch_size)

        if self.h.acyclicity_schedule == "dual":
            self.log('acyclicity/dual', self.dual, batch_size=batch_size)
            self.log('acyclicity/dual_penalty_polyak', self.dual_penalty_polyak, batch_size=batch_size)
        
        # Log SID and example predictions at the first step of every checkpoint_interval
        if step % self.cfg.checkpoint_interval == 0:
            with torch.no_grad():
                # Get predictions for first example in batch
                probs_first = torch.sigmoid(logits)[0].cpu().numpy()
                G_pred_first = (probs_first > self.cfg.decision_threshold).astype(np.float32)
                G_true_first = G_t[0].cpu().numpy()
                
                # Compute SID for first example
                sid_metrics = compute_sid_safe(
                    G_true_first.astype('int8'),
                    G_pred_first.astype('int8'),
                    edge_direction="from row to column"
                )
                
                # Log SID metrics
                self.log('train_checkpoint/sid', sid_metrics['sid'], batch_size=batch_size)
                self.log('train_checkpoint/sid_normalised', sid_metrics['sid_normalised'], batch_size=batch_size)
                self.log('train_checkpoint/is_dag', sid_metrics['is_dag'], batch_size=batch_size)
                
                # Log example prediction/ground truth/probability matrices
                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(
                        key="train_checkpoint/ground_truth",
                        images=[G_true_first],
                        caption=[f"Train Ground Truth (Step {step})"]
                    )
                    self.logger.log_image(
                        key="train_checkpoint/predicted_probs",
                        images=[probs_first],
                        caption=[f"Train Predicted Probabilities (Step {step})"]
                    )
                    self.logger.log_image(
                        key="train_checkpoint/predicted_binary",
                        images=[G_pred_first],
                        caption=[f"Train Predicted Binary (threshold={self.cfg.decision_threshold}, Step {step})"]
                    )
        
        return loss
    
    @override
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, float]:  # type: ignore[override]
        """Validation step - compute metrics on evaluation set."""
        # Extract batch data
        batch_size = len(batch['X'])
        batch_X = batch['X']
        batch_interv_mask = batch['interv_mask']
        G_batch = batch['G']
        
        # Use Polyak-averaged parameters for evaluation
        if self.ave_params_initialized:
            original_params = {name: param.clone() for name, param in self.decoder.named_parameters()}
            for name, param in self.decoder.named_parameters():
                param.data.copy_(self.ave_params[name])
        
        # Forward pass
        logits = self(batch_X, batch_interv_mask)
        probs_batch = torch.sigmoid(logits).cpu().numpy()
        
        # Restore original parameters
        if self.ave_params_initialized:
            for name, param in self.decoder.named_parameters():
                param.data.copy_(original_params[name])
        
        # Compute metrics for each graph in batch
        all_metrics = []
        for b in range(len(batch_X)):
            probs = probs_batch[b]
            G_pred = (probs > self.cfg.decision_threshold).astype(np.float32)
            
            metrics = compute_f1_score(G_batch[b], G_pred)
            
            # Compute SID with graceful cycle handling
            sid_metrics = compute_sid_safe(
                G_batch[b].astype('int8'), 
                G_pred.astype('int8'), 
                edge_direction="from row to column"  # A[i,j]=1 means i→j
            )
            metrics.update(sid_metrics)
            all_metrics.append(metrics)
        
        # Average metrics across batch (use nanmean for SID to handle cycles)
        avg_metrics = {
            'val_precision': float(np.mean([m['precision'] for m in all_metrics])),
            'val_recall': float(np.mean([m['recall'] for m in all_metrics])),
            'val_f1': float(np.mean([m['f1'] for m in all_metrics])),
            'val_sid': float(np.nanmean([m['sid'] for m in all_metrics])),  # nanmean ignores NaN from cycles
            'val_sid_normalised': float(np.nanmean([m['sid_normalised'] for m in all_metrics])),
            'val_is_dag': float(np.mean([m['is_dag'] for m in all_metrics])),  # Fraction of DAGs
        }
        
        # Log metrics
        for key, value in avg_metrics.items():
            self.log(key, value, prog_bar=True, batch_size=batch_size)
        
        # Store first graph's matrices for visualization
        if batch_idx == 0:
            self.last_val_probs = probs_batch[0]
            self.last_val_gt = G_batch[0]
        
        return avg_metrics
    
    @override
    def configure_optimizers(self) -> Union[optim.Optimizer, dict[str, Any]]:  # type: ignore[override]
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.decoder.parameters(),
            lr=self.h.lr,
            weight_decay=self.h.weight_decay
        )
        
        if self.h.lr_schedule == "piecewise" and self.h.lr_drop_steps:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.h.lr_drop_steps,
                gamma=self.h.lr_drop_factor
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }
        else:
            return optimizer


class MatrixVisualizationCallback(Callback):
    """Custom callback for logging adjacency matrices to WandB during validation."""
    
    def __init__(self, cfg: DemoConfig):
        super().__init__()
        self.cfg = cfg
    
    @override
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        """Called after validation ends - log matrices to WandB."""
        if not isinstance(pl_module, CausalDecoderLightningModule):
            return
        
        # Only log if we have matrices from validation
        if not hasattr(pl_module, 'last_val_probs'):
            return
        
        # Log matrices to wandb
        if isinstance(trainer.logger, WandbLogger):
            probs_np = pl_module.last_val_probs
            G_gt = pl_module.last_val_gt
            G_pred_binary = (probs_np > self.cfg.decision_threshold).astype(np.float32)
            
            step = trainer.global_step
            
            # Log images using WandB's native API
            trainer.logger.log_image(
                key="adjacency/ground_truth",
                images=[G_gt],
                caption=[f"Ground Truth (Step {step})"]
            )
            trainer.logger.log_image(
                key="adjacency/predicted_probs",
                images=[probs_np],
                caption=[f"Predicted Probabilities (Step {step})"]
            )
            trainer.logger.log_image(
                key="adjacency/predicted_binary",
                images=[G_pred_binary],
                caption=[f"Predicted Binary (threshold={self.cfg.decision_threshold}, Step {step})"]
            )


# ----------------------------
# Training demo (PyTorch Lightning version)
# ----------------------------

def train_demo(cfg: DemoConfig, h: Hyperparameters) -> dict[str, float]:
    # 1) Setup WandB logger
    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        group=cfg.wandb_group,
        notes=cfg.wandb_notes,
        config={
            "demo": asdict(cfg),
            "hparams": asdict(h),
        },
    )
    
    # Update hyperparameters from wandb.config (for sweeps)
    if hasattr(wandb_logger.experiment.config, 'keys'):
        for key in wandb_logger.experiment.config.keys():
            if hasattr(h, key):
                setattr(h, key, wandb_logger.experiment.config[key])
            elif hasattr(cfg, key):
                setattr(cfg, key, wandb_logger.experiment.config[key])

    # 2) Create training dataset
    num_graphs = h.num_batches * h.batch_size if h.num_batches is not None else None
    train_dataset = CausalGraphDataset(cfg, num_graphs=num_graphs)

    print('Number of graphs:', num_graphs)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=h.batch_size,
        shuffle=False,
        collate_fn=collate_causal_graphs,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 3) Create evaluation dataset (smaller, for checkpointing)
    eval_num_graphs = h.eval_num_batches * h.batch_size
    eval_dataset = CausalGraphDataset(cfg, num_graphs=eval_num_graphs)
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=h.batch_size,
        shuffle=False,
        collate_fn=collate_causal_graphs,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 4) Build backbone
    backbone = build_tabpfn_backbone(cfg)
    emb_dim = int(getattr(backbone, "ninp", h.emsize))
    
    # 5) Create Lightning module
    pl_module = CausalDecoderLightningModule(cfg, h, backbone, emb_dim)
    
    # 6) Create callbacks
    # Built-in checkpoint callback - saves model every N steps
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename='checkpoint_step_{step}',
        every_n_train_steps=cfg.checkpoint_interval,
        save_top_k=-1,  # Save all checkpoints
        verbose=True,
    )
    
    # Custom callback for matrix visualization
    matrix_viz_callback = MatrixVisualizationCallback(cfg)
    
    # 7) Create trainer
    trainer = pl.Trainer(
        max_steps=h.steps,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, matrix_viz_callback],
        gradient_clip_val=h.grad_clip if h.grad_clip > 0 else None,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        val_check_interval=cfg.checkpoint_interval,  # Run validation at checkpoint intervals,
        check_val_every_n_epoch=None,
    )
    
    # 8) Train
    trainer.fit(pl_module, train_dataloader, eval_dataloader)
    
    # 9) Final evaluation
    final_metrics = trainer.validate(pl_module, eval_dataloader, verbose=False)
    
    # Convert to expected format
    metrics = {
        'precision': final_metrics[0]['val_precision'],
        'recall': final_metrics[0]['val_recall'],
        'f1': final_metrics[0]['val_f1'],
        'sid': final_metrics[0]['val_sid'],
        'sid_normalised': final_metrics[0]['val_sid_normalised'],
        'shd': final_metrics[0]['val_shd'],
        'is_dag': final_metrics[0]['val_is_dag'],
        'has_cycles': final_metrics[0]['val_has_cycles'],
    }
    
    return metrics


def run_benchmark(cfg: DemoConfig, h: Hyperparameters) -> None:
    """Run multiple test instances and log results to wandb only."""
    all_metrics = []
    
    for _ in range(cfg.n_test_instances):
        # Run training
        metrics = train_demo(cfg, h)
        all_metrics.append(metrics)
    
    # Compute statistics across all instances
    if len(all_metrics) > 1:
        f1_scores = [m['f1'] for m in all_metrics]
        precision_scores = [m['precision'] for m in all_metrics]
        recall_scores = [m['recall'] for m in all_metrics]
        sid_scores = [m['sid'] for m in all_metrics]
        
        print(f"\n=== Aggregate Results over {cfg.n_test_instances} instances ===")
        print(f"SID: {np.mean(sid_scores):.4f} ± {np.std(sid_scores):.4f}")
        print(f"F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
        print(f"Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")


if __name__ == "__main__":
    # Default configuration
    cfg = DemoConfig()
    h = Hyperparameters()
    
    # Note: For WandB sweeps, hyperparameters will be updated in init_wandb()
    # from wandb.config after wandb.init() is called
    run_benchmark(cfg, h)
