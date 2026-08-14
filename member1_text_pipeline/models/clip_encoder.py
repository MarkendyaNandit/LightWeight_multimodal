"""
CLIP Text Encoder for generating text embeddings.

Loads a frozen CLIP text encoder and provides a clean API for:
  1. Encoding a list of prompt strings into text embeddings.
  2. Encoding all prompts for a given class name.
  3. Projecting CLIP embeddings (512-dim) to match the visual feature
     dimension (768-dim) used by DINO ViT-B/8.

The CLIP encoder is FROZEN — no gradients flow through it.
Only the optional projection layer is trainable.

Based on:
  "Text-Guided Multimodal Unified Industrial Anomaly Detection" (arXiv 2604.22899)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

try:
    import open_clip
except ImportError:
    raise ImportError(
        "open-clip-torch is required. Install it with:\n"
        "  pip install open-clip-torch"
    )

try:
    from .prompt_templates import PromptGenerator
except ImportError:
    from prompt_templates import PromptGenerator


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder with an optional trainable projection layer.

    Takes text prompt strings as input and outputs text embeddings that
    serve as the semantic anchor for the OCTA module.

    Architecture:
        Prompt strings
            → CLIP Tokenizer → Token IDs
            → CLIP Text Transformer (frozen) → Raw embeddings [N × clip_dim]
            → Projection Layer (trainable) → Projected embeddings [N × output_dim]

    Args:
        clip_model_name: Name of the CLIP model variant.
            Default: "ViT-B-32" (512-dim output).
        pretrained: Pretrained weights source. Default: "openai".
        output_dim: Target output dimension for the projection layer.
            Default: 256 (to match 256-dim feature vectors).
            Set to None to skip projection and output raw CLIP embeddings.
        device: Device to load the model on. Default: auto-detect.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        output_dim: int = 256,
        device: Optional[str] = None,
    ):
        super().__init__()

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ── Load CLIP model, tokenizer and image preprocess ──
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        # Freeze the entire CLIP model — no gradients
        self.clip_model.to(device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ── Get CLIP's native embedding dimension ──
        # Encode a dummy text to determine the output dimension
        with torch.no_grad():
            dummy_tokens = self.tokenizer(["test"]).to(device)
            dummy_features = self.clip_model.encode_text(dummy_tokens)
            self.clip_dim = dummy_features.shape[-1]

        # ── Optional projection layer (trainable) ──
        # Projects CLIP's native dim (e.g. 512) to the target dim (e.g. 768)
        # so that text embeddings are compatible with DINO ViT-B/8 visual features.
        self.output_dim = output_dim or self.clip_dim
        if output_dim and output_dim != self.clip_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.clip_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.projection = nn.Identity()

        self.to(device)

    @torch.no_grad()
    def _encode_text_raw(self, text_list: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(text_list).to(self.device)
        text_features = self.clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        img_features = self.clip_model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return self.projection(img_features.float())

    def forward(self, text_list: List[str]) -> torch.Tensor:
        """
        Encode text prompts into projected text embeddings.

        The CLIP encoding is frozen (no gradients), but the projection
        layer is trainable and will receive gradients.

        Args:
            text_list: List of prompt strings.

        Returns:
            Projected text embeddings. Shape: [len(text_list) × output_dim]
        """
        # Step 1: Get raw CLIP embeddings (no grad)
        raw_embeddings = self._encode_text_raw(text_list)

        # Step 2: Project to target dimension (trainable)
        # Detach raw embeddings and re-enable grad for the projection
        projected = self.projection(raw_embeddings.detach())

        return projected

    def encode_class(
        self,
        class_name: str,
        prompt_generator: Optional[PromptGenerator] = None,
    ) -> torch.Tensor:
        """
        Generate and encode all prompts for a given product class.

        Convenience method that combines prompt generation and encoding.

        Args:
            class_name: Product class name (e.g., "mobile phone").
            prompt_generator: PromptGenerator instance. Creates a default one if None.

        Returns:
            Text embeddings for all prompts of this class.
            Shape: [num_prompts × output_dim]

        Example:
            >>> encoder = CLIPTextEncoder()
            >>> embeddings = encoder.encode_class("mobile phone")
            >>> embeddings.shape
            torch.Size([84, 768])
        """
        if prompt_generator is None:
            prompt_generator = PromptGenerator()

        prompts = prompt_generator.generate_prompts(class_name)
        return self.forward(prompts)

    def encode_all_classes(
        self,
        prompt_generator: Optional[PromptGenerator] = None,
    ) -> dict:
        """
        Encode prompts for all registered classes.

        Args:
            prompt_generator: PromptGenerator instance.

        Returns:
            Dict mapping class_name -> embeddings tensor [num_prompts × output_dim].
        """
        if prompt_generator is None:
            prompt_generator = PromptGenerator()

        return {
            cls: self.encode_class(cls, prompt_generator)
            for cls in prompt_generator.get_classes()
        }

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim

    def get_clip_dim(self) -> int:
        """Return the native CLIP embedding dimension."""
        return self.clip_dim

    def extra_repr(self) -> str:
        return (
            f"clip_dim={self.clip_dim}, "
            f"output_dim={self.output_dim}, "
            f"device={self.device}"
        )


# ──────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading CLIP Text Encoder...")
    encoder = CLIPTextEncoder(output_dim=768)
    print(f"  CLIP dim: {encoder.get_clip_dim()}")
    print(f"  Output dim: {encoder.get_embedding_dim()}")

    # Check frozen vs trainable parameters
    frozen = sum(p.numel() for p in encoder.clip_model.parameters())
    trainable = sum(
        p.numel() for p in encoder.projection.parameters() if p.requires_grad
    )
    print(f"  Frozen params (CLIP): {frozen:,}")
    print(f"  Trainable params (projection): {trainable:,}")

    # Encode a single class
    gen = PromptGenerator()
    embeddings = encoder.encode_class("mobile phone", gen)
    print(f"\n'mobile phone' embeddings shape: {embeddings.shape}")
    print(f"  requires_grad: {embeddings.requires_grad}")
