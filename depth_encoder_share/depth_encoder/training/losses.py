"""
training/losses.py

VICReg loss function.

VICReg = Variance + Invariance + Covariance Regularisation.

Given two expanded representation matrices Z1, Z2 ∈ ℝ^(B × D):

  L_total = λ · L_invariance
           + μ · (L_variance(Z1) + L_variance(Z2))
           + ν · (L_covariance(Z1) + L_covariance(Z2))

Terms
-----
L_invariance  (Sim):
  MSE between Z1 and Z2.  Attracts representations of the same image.

L_variance  (Std):
  Hinge loss: max(0, 1 − std(z_j)) summed over feature dimensions j.
  Prevents feature collapse by keeping each dimension's std ≥ 1 across
  the batch.

L_covariance  (Cov):
  Sum of squared off-diagonal entries of the feature covariance matrix,
  normalised by D.  Decorrelates dimensions, maximising information.

Reference:
  Bardes et al., VICReg, ICLR 2022. https://arxiv.org/abs/2105.04906
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VICRegLossOutput:
    """Decomposed VICReg loss for logging."""
    total:      torch.Tensor
    invariance: torch.Tensor
    variance:   torch.Tensor
    covariance: torch.Tensor


# ---------------------------------------------------------------------------
# Component losses
# ---------------------------------------------------------------------------

def _invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between Z1 and Z2 (no gradient stop)."""
    return F.mse_loss(z1, z2)


def _variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Hinge loss that pushes per-dimension std of the batch to ≥ 1.

    z : (B, D)
    """
    std = torch.sqrt(z.var(dim=0) + eps)          # (D,)
    loss = F.relu(1.0 - std).mean()
    return loss


def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Off-diagonal covariance penalty.

    Encourages different dimensions to be de-correlated across the batch.

    z : (B, D)
    """
    B, D = z.shape
    z = z - z.mean(dim=0)                         # centre

    cov = (z.T @ z) / (B - 1)                     # (D, D)

    # Mask diagonal
    off_diag = cov.pow(2)
    off_diag.fill_diagonal_(0.0)

    loss = off_diag.sum() / D
    return loss


# ---------------------------------------------------------------------------
# VICReg loss (main entry point)
# ---------------------------------------------------------------------------

def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> VICRegLossOutput:
    """
    Compute the full VICReg loss.

    Parameters
    ----------
    z1, z2 : torch.Tensor
        Expanded representations of two augmented views.
        Shape (B, D) — output of the Expander MLP.
    sim_coeff : float
        Weight for the invariance (similarity) term (λ).
    std_coeff : float
        Weight for the variance term (μ).
    cov_coeff : float
        Weight for the covariance term (ν).
    eps : float
        Small constant added inside the variance std computation.

    Returns
    -------
    VICRegLossOutput
        Named dataclass with .total, .invariance, .variance, .covariance.
    """
    inv_loss  = _invariance_loss(z1, z2)
    var_loss  = _variance_loss(z1, eps) + _variance_loss(z2, eps)
    cov_loss  = _covariance_loss(z1) + _covariance_loss(z2)

    total = (
        sim_coeff * inv_loss
        + std_coeff * var_loss
        + cov_coeff * cov_loss
    )

    return VICRegLossOutput(
        total=total,
        invariance=inv_loss,
        variance=var_loss,
        covariance=cov_loss,
    )
