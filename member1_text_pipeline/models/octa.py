"""
Object-Conditioned Textual Feature Adaptor (OCTA).

This module adapts static text embeddings from the CLIP encoder into
robust visual-linguistic anchors using a Mixture-of-Experts (MoE)
architecture and a Prototype-based Cross-Attention mechanism.

Based on:
  "Text-Guided Multimodal Unified Industrial Anomaly Detection" (arXiv 2604.22899)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Expert(nn.Module):
    """
    A single expert network in the MoE module.
    Implemented as a 2-layer MLP with GELU activation.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKMoE(nn.Module):
    """
    Top-K Mixture-of-Experts module.
    
    Routes input text embeddings through specialized expert networks
    to capture diverse semantic nuances.
    """
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Gating network to compute routing probabilities
        self.gate = nn.Linear(dim, num_experts)
        
        # Expert networks
        hidden_dim = dim * 4
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input text embeddings [N × D]
        Returns:
            Enhanced text embeddings [N × D]
        """
        # Calculate routing probabilities [N × num_experts]
        logits = self.gate(x)
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Output tensor initialized to zeros
        out = torch.zeros_like(x)
        
        # For each expert in the top-k, route the inputs and accumulate
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_gates = top_k_gates[:, i].unsqueeze(-1)
            
            # Since experts can be different for each batch element,
            # we iterate over the batch or use gather operations.
            # For simplicity and correctness across batch elements:
            for expert_idx in range(self.num_experts):
                # Find which items in the batch are routed to this expert
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    out[mask] += expert_output * expert_gates[mask]
                    
        return out


class PrototypeCrossAttention(nn.Module):
    """
    Cross-Attention mechanism with learnable prototypes.
    
    Query = Learnable prototypes
    Key, Value = Enhanced text embeddings
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = math.sqrt(dim)
        
        # Learnable prototype (Query)
        # The paper mentions P \in R^{1 \times D}
        self.prototype = nn.Parameter(torch.randn(1, 1, dim))
        
        # Projections
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeddings: Enhanced text embeddings [N × D]
        Returns:
            Attended features [1 × D]
        """
        # text_embeddings is [N, D] -> reshape to [1, N, D] for attention
        x = text_embeddings.unsqueeze(0)
        
        # Q: [1, 1, D]
        # K, V: [1, N, D]
        q = self.W_q(self.prototype)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Attention scores: [1, 1, N]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Output: [1, 1, D] -> [1, D]
        out = torch.matmul(attn_probs, v)
        return out.squeeze(0)


class OCTA(nn.Module):
    """
    Object-Conditioned Textual Feature Adaptor (OCTA).
    
    Combines Top-K MoE, Prototype Cross-Attention, and nonlinear transformation.
    Produces a class-specific adapted text feature (F_p) that serves as the 
    semantic anchor for the visual modalities.
    """
    def __init__(self, dim: int = 256, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.dim = dim
        
        # 1. Top-K MoE
        self.moe = TopKMoE(dim=dim, num_experts=num_experts, top_k=top_k)
        
        # 2. Prototype Cross-Attention
        self.cross_attention = PrototypeCrossAttention(dim=dim)
        
        # 3. Nonlinear Transformation
        self.nonlinear_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeddings: Initial text embeddings from CLIP [N × D]
        Returns:
            Adapted text features F_p [1 × D]
        """
        # 1. Enhance text embeddings using MoE
        t_enhanced = self.moe(text_embeddings)
        
        # 2. Cross-Attention with learnable prototype
        f_attended = self.cross_attention(t_enhanced)
        
        # 3. Nonlinear transformation
        f_p = self.nonlinear_transform(f_attended)
        
        return f_p

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ──────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading OCTA Module...")
    octa = OCTA(dim=256)
    
    # Dummy text embeddings (e.g., 84 prompts for a class)
    N = 84
    dummy_text_embeds = torch.randn(N, 256)
    
    print(f"Input text embeddings shape: {dummy_text_embeds.shape}")
    
    # Forward pass
    f_p = octa(dummy_text_embeds)
    
    print(f"Adapted text features (F_p) shape: {f_p.shape}")
    
    trainable = sum(p.numel() for p in octa.parameters() if p.requires_grad)
    print(f"Trainable params (OCTA): {trainable:,}")
