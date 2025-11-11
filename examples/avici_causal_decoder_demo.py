import random
from dataclasses import dataclass, asdict
from typing import Optional, Any, Callable, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from cdt.metrics import SID
import tqdm

# Provide a local no-op @override decorator recognized by type checkers
Fn = TypeVar("Fn", bound=Callable[..., Any])
def override(func: Fn) -> Fn:  # type: ignore[misc]
    return func

# (No direct need for get_architecture/ModelConfig when loading pretrained)
from tabpfn.model_loading import load_model_criterion_config

# Weights & Biases logging (assumed installed and always enabled)
import wandb

# Global wandb run and base-step offset for logging across instances
WB_RUN: Optional[wandb.Run] = None
WB_BASE_STEP_OFFSET: int = 0


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
    # Evaluation
    n_test_instances: int = 1  # number of test instances to average over
    decision_threshold: float = 0.5  # threshold for converting probabilities to binary predictions
    # Always load official TabPFN-v2 classifier weights; no fallback to random init
    model_path: Optional[str] = None  # keep None to auto-download/cache

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
    lr: float = 3e-5  # AVICI LINEAR base learning rate
    weight_decay: float = 0.0  # AVICI default (LAMB optimizer handles weight decay)
    steps: int = 100000  # Increased to 100k (AVICI uses 300k, but 100k for faster testing)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8  # Number of graphs per batch
    # DataLoader configuration (cache-on-the-fly mode)
    num_batches: Optional[int] = None  # Max batches to cache (None = infinite, never caches)
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
    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
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
                print(f"✓ Completed pass through cache. Reshuffling {len(self.graphs)} graphs.")
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


def init_wandb(cfg: DemoConfig, h: Hyperparameters):
    """Initialize a Weights & Biases run (always enabled) and store it globally.
    
    For WandB sweeps, this will automatically use sweep configuration.
    """
    global WB_RUN
    WB_RUN = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        group=cfg.wandb_group,
        notes=cfg.wandb_notes,
        config={
            "demo": asdict(cfg),
            "hparams": asdict(h),
        },
    )
    
    # For WandB sweeps: Update hyperparameters from wandb.config
    # This allows sweep to override default values
    if WB_RUN is not None and hasattr(wandb.config, 'keys'):
        # Update hyperparameters from sweep config
        for key in wandb.config.keys():
            if hasattr(h, key):
                setattr(h, key, wandb.config[key])
            elif hasattr(cfg, key):
                setattr(cfg, key, wandb.config[key])


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

def train_demo(cfg: DemoConfig, h: Hyperparameters) -> dict[str, float]:
    # 1) Create PyTorch Dataset (always cache-on-the-fly)
    num_graphs = h.num_batches * h.batch_size if h.num_batches is not None else None
    dataset = CausalGraphDataset(cfg, num_graphs=num_graphs)
    
    # Create PyTorch DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=h.batch_size,
        shuffle=False,  # Don't shuffle to maintain cache order
        collate_fn=collate_causal_graphs,
        num_workers=0,  # Use 0 for now (main process), can increase for parallel generation
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create iterator
    data_iter = iter(dataloader)

    # 2) Backbone
    model = build_tabpfn_backbone(cfg).to(h.device)
    model.eval()  # freeze backbone for the demo
    for p in model.parameters():
        p.requires_grad_(False)
    
    # No console logging

    # 3) Decoder — match the backbone's actual embedding size
    emb_dim = int(getattr(model, "ninp", h.emsize))
    decoder = CausalGraphDecoder(emb_dim).to(h.device)
    opt = optim.AdamW(decoder.parameters(), lr=h.lr, weight_decay=h.weight_decay)
    
    # Learning rate scheduler (AVICI uses piecewise constant)
    if h.lr_schedule == "piecewise" and h.lr_drop_steps:
        # Create milestones dict for MultiStepLR
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=h.lr_drop_steps, gamma=h.lr_drop_factor)
    else:
        scheduler = None
    
    # Initialize Polyak-averaged parameters (EMA)
    ave_params = {name: param.clone().detach() for name, param in decoder.named_parameters()}

    # 4) Initialize dual variable and Polyak-averaged penalty for dual acyclicity scheduling
    dual = torch.tensor(0.0, device=h.device, dtype=torch.float32)
    dual_penalty_polyak = torch.tensor(0.0, device=h.device, dtype=torch.float32)

    # 5) Train loop (decoder only) - iterate over DataLoader
    for step in tqdm.trange(h.steps):
        # Get next batch from DataLoader
        try:
            batch = next(data_iter)
        except StopIteration:
            # For pregenerated mode, restart iterator when exhausted
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Extract batch data
        batch_X = batch['X']  # list of [n_obs, n_vars]
        batch_interv_mask = batch['interv_mask']  # list of [n_obs, n_vars]
        G_batch = batch['G']  # [batch_size, n_vars, n_vars]
        
        # Prepare ground truth tensor
        G_t = torch.as_tensor(G_batch, device=h.device, dtype=torch.float32)
        
        opt.zero_grad(set_to_none=True)
        
        # Get node embeddings for all graphs in batch (parallel processing via TabPFN's batch dim)
        node_emb_batch = get_node_embeddings_from_tabpfn_batched(
            model, batch_X, batch_interv_mask
        )  # [batch_size, n_vars, E]
        logits = decoder(node_emb_batch)  # [batch_size, n_vars, n_vars]
        
        # AVICI-style BCE with label smoothing and positive weighting
        diag_mask = torch.eye(logits.size(-1), device=logits.device, dtype=torch.bool)
        n_vars = logits.size(-1)
        
        # Apply label smoothing to targets
        y_soft = (1 - h.label_smoothing) * G_t + h.label_smoothing / 2.0
        
        # Compute log probabilities
        logp1 = F.logsigmoid(logits)
        logp0 = F.logsigmoid(-logits)
        
        # Binary cross-entropy with positive weighting
        xent_eltwise = -(h.pos_weight * y_soft * logp1 + (1 - y_soft) * logp0)
        
        # Mask diagonal and normalize by number of edges (AVICI style)
        xent_masked = xent_eltwise.masked_fill(diag_mask.unsqueeze(0), 0.0)
        # Average over batch, sum over edges, normalize by d(d-1)
        bce = xent_masked.sum(dim=(-2, -1)).mean() / (n_vars * (n_vars - 1))
        
        # Acyclicity penalty in log-space (AVICI style)
        acyc_penalty = acyclicity_penalty_from_logits(logits, iters=h.power_iters).mean()
        
        # Adaptive acyclicity weight scheduling (matching AVICI)
        if h.acyclicity_schedule == "const":
            acyc_weight = h.acyclicity_weight
        elif h.acyclicity_schedule == "linear":
            acyc_weight = h.acyclicity_weight * max(0.0, (step - h.acyclicity_burnin) * h.acyclicity_linear_rate)
        elif h.acyclicity_schedule == "dual":
            # Dual ascent (augmented Lagrangian) - AVICI style
            acyc_weight = h.acyclicity_weight * dual.item()
        else:
            raise ValueError(f"Unknown acyclicity schedule: {h.acyclicity_schedule}")
        
        wgt_acyc = acyc_weight * acyc_penalty
        loss = bce + wgt_acyc
        loss.backward()
        
        # Gradient clipping (AVICI default)
        if h.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), h.grad_clip)
        
        opt.step()
        
        # Update Polyak-averaged parameters (EMA) - AVICI style
        with torch.no_grad():
            for name, param in decoder.named_parameters():
                ave_params[name].mul_(1 - h.polyak_step_size).add_(param, alpha=h.polyak_step_size)
        
        # Update dual variable (for dual schedule) - AVICI style
        if h.acyclicity_schedule == "dual":
            with torch.no_grad():
                # Polyak averaging of acyclicity penalty (AVICI uses this for stability)
                if step == 0:
                    dual_penalty_polyak = acyc_penalty.clone()
                else:
                    dual_penalty_polyak = (1 - h.acyclicity_polyak) * dual_penalty_polyak + h.acyclicity_polyak * acyc_penalty
                
                # Dual learning rate with warmup (AVICI gradually increases dual_lr)
                if h.acyclicity_warmup:
                    dual_lr = min(h.acyclicity_dual_lr, step * h.acyclicity_dual_lr / h.acyclicity_burnin)
                    effective_burnin = 0  # warmup replaces burnin
                else:
                    dual_lr = h.acyclicity_dual_lr
                    effective_burnin = h.acyclicity_burnin
                
                # Update dual every inner_step iterations after burnin
                if (step % h.acyclicity_inner_step == 0) and (step > effective_burnin):
                    dual = dual + dual_lr * dual_penalty_polyak
        
        # Learning rate schedule step
        if scheduler is not None:
            scheduler.step()

        # Log to wandb every step (no console logging)
        if WB_RUN is not None:
            with torch.no_grad():
                # Use first element in batch for visualization metrics
                probs = torch.sigmoid(logits)[0]
                pos_mae = (probs[G_t[0] > 0] - 1.0).abs().mean().item() if (G_t[0] > 0).any() else 0.0
                neg_mae = (probs[G_t[0] == 0] - 0.0).abs().mean().item()
            log_payload = {
                "loss": float(loss.item()),
                "bce": float(bce.item()),
                "acyclicity/raw": float(acyc_penalty.item()),
                "acyclicity/weight": float(acyc_weight),
                "acyclicity/weighted": float(wgt_acyc.item()),
                "metrics/pos_mae": float(pos_mae),
                "metrics/neg_mae": float(neg_mae),
                "hp/lr": float(opt.param_groups[0]['lr']),  # Log actual LR (accounts for schedule)
                "hp/weight_decay": float(h.weight_decay),
                "step": int(WB_BASE_STEP_OFFSET + step + 1),
            }
            if h.acyclicity_schedule == "dual":
                log_payload["acyclicity/dual"] = float(dual.item())
                log_payload["acyclicity/dual_penalty_polyak"] = float(dual_penalty_polyak.item())
                if h.acyclicity_warmup:
                    current_dual_lr = min(h.acyclicity_dual_lr, step * h.acyclicity_dual_lr / h.acyclicity_burnin)
                    log_payload["acyclicity/dual_lr"] = float(current_dual_lr)
            
            # Log adjacency matrices every 1000 steps (first element in batch)
            if (step + 1) % 1000 == 0:
                with torch.no_grad():
                    probs_np = probs.cpu().numpy()
                    G_pred_binary = (probs_np > cfg.decision_threshold).astype(np.float32)
                    
                    # Log matrices directly as wandb Images (first element in batch)
                    # wandb.Image accepts numpy arrays and will render them as heatmaps
                    log_payload["adjacency/ground_truth"] = wandb.Image(
                        G_batch[0],  # First graph in batch
                        caption=f"Ground Truth [0] (Step {step + 1})"
                    )
                    log_payload["adjacency/predicted_probs"] = wandb.Image(
                        probs_np,
                        caption=f"Predicted Probabilities [0] (Step {step + 1})"
                    )
                    log_payload["adjacency/predicted_binary"] = wandb.Image(
                        G_pred_binary,
                        caption=f"Predicted Binary [0] (threshold={cfg.decision_threshold}, Step {step + 1})"
                    )
            
            cast(Any, WB_RUN).log(log_payload, step=WB_BASE_STEP_OFFSET + step + 1)
    
    # 6) Final evaluation (use Polyak-averaged parameters for better stability)
    # Generate a fresh batch for evaluation using the dataset
    eval_dataset = CausalGraphDataset(cfg, num_graphs=h.batch_size)
    eval_batch_list = [eval_dataset[i] for i in range(h.batch_size)]
    eval_batch = collate_causal_graphs(eval_batch_list)
    eval_batch_X = eval_batch['X']
    eval_batch_interv_mask = eval_batch['interv_mask']
    eval_G_batch = eval_batch['G']
    
    with torch.no_grad():
        # Temporarily load Polyak-averaged parameters
        original_params = {name: param.clone() for name, param in decoder.named_parameters()}
        for name, param in decoder.named_parameters():
            param.data.copy_(ave_params[name])
        
        # Get embeddings for all graphs in evaluation batch
        node_emb_batch = get_node_embeddings_from_tabpfn_batched(
            model, eval_batch_X, eval_batch_interv_mask
        )  # [batch_size, n_vars, E]
        logits = decoder(node_emb_batch)  # [batch_size, n_vars, n_vars]
        probs_batch = torch.sigmoid(logits).cpu().numpy()  # [batch_size, n_vars, n_vars]
        
        # Restore original parameters
        for name, param in decoder.named_parameters():
            param.data.copy_(original_params[name])
        
        # Compute metrics for each graph in batch
        all_batch_metrics = []
        for b in range(h.batch_size):
            probs = probs_batch[b]
            G_pred = (probs > cfg.decision_threshold).astype(np.float32)
            
            batch_metrics = compute_f1_score(eval_G_batch[b], G_pred)
            batch_metrics['sid'] = SID(eval_G_batch[b], G_pred)
            all_batch_metrics.append(batch_metrics)
        
        # Average metrics across batch
        metrics = {
            'precision': np.mean([m['precision'] for m in all_batch_metrics]),
            'recall': np.mean([m['recall'] for m in all_batch_metrics]),
            'f1': np.mean([m['f1'] for m in all_batch_metrics]),
            'sid': np.mean([m['sid'] for m in all_batch_metrics]),
            'tp': np.mean([m['tp'] for m in all_batch_metrics]),
            'fp': np.mean([m['fp'] for m in all_batch_metrics]),
            'fn': np.mean([m['fn'] for m in all_batch_metrics]),
        }
    
    # No console logging
        
        # Final logging to wandb (summary - averaged over batch)
        if WB_RUN is not None:
            cast(Any, WB_RUN).log({
                "final/precision": float(metrics['precision']),
                "final/recall": float(metrics['recall']),
                "final/f1": float(metrics['f1']),
                "final/sid": float(metrics['sid']),
                "final/tp": float(metrics['tp']),
                "final/fp": float(metrics['fp']),
                "final/fn": float(metrics['fn']),
            }, step=WB_BASE_STEP_OFFSET + h.steps)
    return metrics


def run_benchmark(cfg: DemoConfig, h: Hyperparameters) -> None:
    """Run multiple test instances and log results to wandb only."""
    # no console logging

    all_metrics = []
    # Initialize wandb once per benchmark run
    init_wandb(cfg, h)
    
    for instance in range(cfg.n_test_instances):
        # # Use different seed for each instance
        # set_seed(instance * 100)
        
        base_step = instance * h.steps
        # update global base-step offset for logging within this instance
        global WB_BASE_STEP_OFFSET
        WB_BASE_STEP_OFFSET = base_step
        metrics = train_demo(cfg, h)
        all_metrics.append(metrics)
    # no per-instance console output
    
    # Compute statistics across all instances
    f1_scores = [m['f1'] for m in all_metrics]
    precision_scores = [m['precision'] for m in all_metrics]
    recall_scores = [m['recall'] for m in all_metrics]
    sid_scores = [m['sid'] for m in all_metrics]
    # no final console output

    # Log aggregate results to wandb and close run
    if WB_RUN is not None:
        cast(Any, WB_RUN).log({
            "aggregate/SID_mean": float(np.mean(sid_scores)),
            "aggregate/SID_std": float(np.std(sid_scores)),
            "aggregate/F1_mean": float(np.mean(f1_scores)),
            "aggregate/F1_std": float(np.std(f1_scores)),
            "aggregate/Precision_mean": float(np.mean(precision_scores)),
            "aggregate/Precision_std": float(np.std(precision_scores)),
            "aggregate/Recall_mean": float(np.mean(recall_scores)),
            "aggregate/Recall_std": float(np.std(recall_scores)),
        })
        cast(Any, WB_RUN).finish()


if __name__ == "__main__":
    # Default configuration
    cfg = DemoConfig()
    h = Hyperparameters()
    
    # Note: For WandB sweeps, hyperparameters will be updated in init_wandb()
    # from wandb.config after wandb.init() is called
    run_benchmark(cfg, h)
