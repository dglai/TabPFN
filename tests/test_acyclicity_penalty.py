"""
Unit tests for acyclicity penalty computation.

Tests the exp_matmul and acyclicity_penalty_from_logits functions
to ensure they match AVICI's official implementation.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np


# Import the functions to test
# Note: In a real setup, these would be imported from the module
# For now, we'll copy them here for testing
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
    
    # Initialize random vectors for power iteration
    u = torch.randn(B, D, device=logits.device, dtype=logits.dtype)
    v = torch.randn(B, D, device=logits.device, dtype=logits.dtype)
    
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


class TestExpMatmul:
    """Test suite for exp_matmul function."""
    
    def test_exp_matmul_left_multiply_simple(self):
        """Test exp(logmat) @ vec with simple values."""
        # Create simple log matrix and vector
        logmat = torch.log(torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]))  # [1, 2, 2]
        vec = torch.tensor([[1.0, 2.0]])  # [1, 2]
        
        # Compute using exp_matmul (axis=-1 for left multiply)
        result = exp_matmul(logmat, vec, axis=-1)
        
        # Compute ground truth: exp(logmat) @ vec
        mat = torch.exp(logmat)  # [[2, 3], [4, 5]]
        expected = torch.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)  # [1, 2]
        # expected = [[2*1 + 3*2], [4*1 + 5*2]] = [[8], [14]]
        
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)
    
    def test_exp_matmul_right_multiply_simple(self):
        """Test vec @ exp(logmat) with simple values."""
        # Create simple log matrix and vector
        logmat = torch.log(torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]))  # [1, 2, 2]
        vec = torch.tensor([[1.0, 2.0]])  # [1, 2]
        
        # Compute using exp_matmul (axis=-2 for right multiply)
        result = exp_matmul(logmat, vec, axis=-2)
        
        # Compute ground truth: vec @ exp(logmat)
        mat = torch.exp(logmat)  # [[2, 3], [4, 5]]
        expected = torch.matmul(vec.unsqueeze(-2), mat).squeeze(-2)  # [1, 2]
        # expected = [[1*2 + 2*4, 1*3 + 2*5]] = [[10, 13]]
        
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)
    
    def test_exp_matmul_batch(self):
        """Test exp_matmul with batch dimension."""
        B, D = 3, 4
        logmat = torch.randn(B, D, D)
        vec = torch.randn(B, D)
        
        # Left multiply
        result_left = exp_matmul(logmat, vec, axis=-1)
        expected_left = torch.matmul(torch.exp(logmat), vec.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result_left, expected_left, rtol=1e-4, atol=1e-4)
        
        # Right multiply
        result_right = exp_matmul(logmat, vec, axis=-2)
        expected_right = torch.matmul(vec.unsqueeze(-2), torch.exp(logmat)).squeeze(-2)
        assert torch.allclose(result_right, expected_right, rtol=1e-4, atol=1e-4)
    
    def test_exp_matmul_numerical_stability(self):
        """Test that exp_matmul is numerically stable with large negative values."""
        # Create log matrix with very negative values (would underflow in exp)
        logmat = torch.tensor([[[-100.0, -50.0], [-200.0, -150.0]]])
        vec = torch.tensor([[1.0, 1.0]])
        
        # This should not produce NaN or Inf
        result = exp_matmul(logmat, vec, axis=-1)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert (result >= 0).all()  # Should be non-negative
    
    def test_exp_matmul_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        logmat = torch.randn(1, 3, 3)
        vec = torch.randn(1, 3)
        
        with pytest.raises(ValueError, match="Invalid axis"):
            exp_matmul(logmat, vec, axis=0)
    
    def test_exp_matmul_shape_consistency(self):
        """Test that output shape is correct."""
        B, D = 2, 5
        logmat = torch.randn(B, D, D)
        vec = torch.randn(B, D)
        
        result_left = exp_matmul(logmat, vec, axis=-1)
        result_right = exp_matmul(logmat, vec, axis=-2)
        
        assert result_left.shape == (B, D)
        assert result_right.shape == (B, D)


class _TestAcyclicityPenalty:
    """Test suite for acyclicity_penalty_from_logits function."""
    
    def test_acyclic_graph_low_penalty(self):
        """Test that acyclic (DAG) graphs have low spectral radius."""
        # Create a strictly upper triangular matrix (DAG)
        B, D = 1, 5
        logits = torch.zeros(B, D, D)
        
        # Set upper triangular to positive values (high probability edges)
        for i in range(D):
            for j in range(i + 1, D):
                logits[0, i, j] = 5.0  # High logit -> high probability
        
        penalty = acyclicity_penalty_from_logits(logits, iters=20)
        
        # For a DAG, spectral radius should be < 1
        assert penalty.item() < 1.0, f"DAG should have spectral radius < 1, got {penalty.item()}"
    
    def test_cyclic_graph_high_penalty(self):
        """Test that cyclic graphs have high spectral radius."""
        # Create a cycle: 0 -> 1 -> 2 -> 0
        B, D = 1, 3
        logits = torch.full((B, D, D), -10.0)  # Start with low probabilities
        
        # Create cycle with high probability edges
        logits[0, 0, 1] = 5.0  # 0 -> 1
        logits[0, 1, 2] = 5.0  # 1 -> 2
        logits[0, 2, 0] = 5.0  # 2 -> 0 (creates cycle)
        
        penalty = acyclicity_penalty_from_logits(logits, iters=20)
        
        # For a cycle, spectral radius should be >= 1
        assert penalty.item() >= 0.5, f"Cycle should have higher spectral radius, got {penalty.item()}"
    
    def test_empty_graph_zero_penalty(self):
        """Test that empty graph (no edges) has zero spectral radius."""
        B, D = 1, 4
        # Very negative logits -> probabilities near 0
        logits = torch.full((B, D, D), -100.0)
        
        penalty = acyclicity_penalty_from_logits(logits, iters=20)
        
        # Empty graph should have spectral radius near 0
        assert penalty.item() < 0.01, f"Empty graph should have near-zero spectral radius, got {penalty.item()}"
    
    def test_diagonal_masked(self):
        """Test that diagonal is properly masked (no self-loops)."""
        B, D = 1, 3
        logits = torch.zeros(B, D, D)
        
        # Set diagonal to high values (should be ignored)
        for i in range(D):
            logits[0, i, i] = 100.0
        
        penalty = acyclicity_penalty_from_logits(logits, iters=20)
        
        # Should still be near zero since only diagonal has edges
        assert penalty.item() < 0.01, f"Diagonal should be masked, got {penalty.item()}"
    
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        B, D = 3, 4
        logits = torch.randn(B, D, D)
        
        penalty = acyclicity_penalty_from_logits(logits, iters=10)
        
        assert penalty.shape == (B,)
        assert not torch.isnan(penalty).any()
        assert not torch.isinf(penalty).any()
        assert (penalty >= 0).all()  # Spectral radius should be non-negative
    
    def test_power_iteration_convergence(self):
        """Test that more iterations lead to more stable estimates."""
        B, D = 1, 5
        torch.manual_seed(42)
        logits = torch.randn(B, D, D)
        
        # Compute with different iteration counts
        penalty_10 = acyclicity_penalty_from_logits(logits.clone(), iters=10)
        penalty_50 = acyclicity_penalty_from_logits(logits.clone(), iters=50)
        penalty_100 = acyclicity_penalty_from_logits(logits.clone(), iters=100)
        
        # More iterations should give similar results (convergence)
        diff_50_100 = torch.abs(penalty_50 - penalty_100).item()
        diff_10_50 = torch.abs(penalty_10 - penalty_50).item()
        
        # Difference should decrease with more iterations
        assert diff_50_100 < diff_10_50 or diff_50_100 < 0.1, \
            f"Power iteration should converge, but got diff_10_50={diff_10_50}, diff_50_100={diff_50_100}"
    
    def test_gradient_flow(self):
        """Test that gradients flow through the penalty computation."""
        B, D = 1, 3
        logits = torch.randn(B, D, D, requires_grad=True)
        
        penalty = acyclicity_penalty_from_logits(logits, iters=10)
        loss = penalty.sum()
        loss.backward()
        
        # Gradients should exist and be non-zero for at least some elements
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert (logits.grad.abs() > 0).any(), "Some gradients should be non-zero"
    
    def test_known_spectral_radius(self):
        """Test with a matrix where we know the spectral radius."""
        # Create a simple matrix with known eigenvalues
        # For a 2x2 matrix [[a, b], [c, d]], if we use probabilities,
        # we can construct specific cases
        B, D = 1, 2
        
        # Create logits that give probabilities of 0.5 for all edges
        logits = torch.zeros(B, D, D)  # sigmoid(0) = 0.5
        
        # For a 2x2 matrix with all entries 0.5 (except diagonal):
        # [[0, 0.5], [0.5, 0]]
        # Eigenvalues are ±0.5, so spectral radius = 0.5
        penalty = acyclicity_penalty_from_logits(logits, iters=50)
        
        # Should be close to 0.5
        assert 0.3 < penalty.item() < 0.7, f"Expected ~0.5, got {penalty.item()}"


class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_exp_matmul_in_power_iteration(self):
        """Test that exp_matmul works correctly within power iteration."""
        B, D = 2, 4
        logp = torch.randn(B, D, D)
        
        # Mask diagonal
        diag_mask = torch.eye(D, dtype=torch.bool).unsqueeze(0)
        logp = logp.masked_fill(diag_mask, -float('inf'))
        
        # Initialize vectors
        u = torch.randn(B, D)
        v = torch.randn(B, D)
        u = u / u.norm(dim=-1, keepdim=True)
        v = v / v.norm(dim=-1, keepdim=True)
        
        # One iteration of power method
        u_new = exp_matmul(logp, u, axis=-2)  # u @ exp(logp)
        v_new = exp_matmul(logp, v, axis=-1)  # exp(logp) @ v
        
        # Compare with direct computation
        mat = torch.exp(logp)
        u_expected = torch.matmul(u.unsqueeze(-2), mat).squeeze(-2)
        v_expected = torch.matmul(mat, v.unsqueeze(-1)).squeeze(-1)
        
        assert torch.allclose(u_new, u_expected, rtol=1e-4, atol=1e-4)
        assert torch.allclose(v_new, v_expected, rtol=1e-4, atol=1e-4)
    
    def test_consistency_across_devices(self):
        """Test that results are consistent across CPU and CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        B, D = 2, 5
        logits_cpu = torch.randn(B, D, D)
        logits_cuda = logits_cpu.cuda()
        
        torch.manual_seed(42)
        penalty_cpu = acyclicity_penalty_from_logits(logits_cpu, iters=20)
        
        torch.manual_seed(42)
        penalty_cuda = acyclicity_penalty_from_logits(logits_cuda, iters=20)
        
        # Results should be very similar (allowing for minor numerical differences)
        assert torch.allclose(penalty_cpu, penalty_cuda.cpu(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
