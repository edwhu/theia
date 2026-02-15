# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class ViTPatchEmbedding(nnx.Module):
    """Patch embedding using a convolution with kernel_size=stride=patch_size."""

    def __init__(self, hidden_size: int, patch_size: int, num_channels: int = 3, *, rngs: nnx.Rngs):
        self.projection = nnx.Conv(
            in_features=num_channels,
            out_features=hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, pixel_values: jax.Array) -> jax.Array:
        # pixel_values: [B, C, H, W] -> channels-last [B, H, W, C]
        x = jnp.transpose(pixel_values, (0, 2, 3, 1))
        x = self.projection(x)  # [B, H', W', hidden_size]
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c)  # [B, num_patches, hidden_size]


class ViTEmbeddings(nnx.Module):
    """Patch embedding + CLS token + positional embeddings."""

    def __init__(self, hidden_size: int, patch_size: int, num_patches: int, num_channels: int = 3, *, rngs: nnx.Rngs):
        self.patch_embeddings = ViTPatchEmbedding(hidden_size, patch_size, num_channels, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.position_embeddings = nnx.Param(jnp.zeros((1, 1 + num_patches, hidden_size)))

    def __call__(self, pixel_values: jax.Array) -> jax.Array:
        b = pixel_values.shape[0]
        patch_embeds = self.patch_embeddings(pixel_values)  # [B, num_patches, hidden_size]
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (b, 1, patch_embeds.shape[-1]))
        embeddings = jnp.concatenate([cls_tokens, patch_embeds], axis=1)  # [B, 1+num_patches, hidden_size]
        embeddings = embeddings + self.position_embeddings.value
        return embeddings


class ViTAttention(nnx.Module):
    """Multi-head self-attention with separate Q/K/V projections."""

    def __init__(self, hidden_size: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.key = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.value = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.output_dense = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        b, n, _ = x.shape
        q = self.query(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(b, n, -1)
        return self.output_dense(attn_output)


class ViTMLP(nnx.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(hidden_size, intermediate_size, rngs=rngs)
        self.dense2 = nnx.Linear(intermediate_size, hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense2(jax.nn.gelu(self.dense1(x), approximate=False))


class ViTBlock(nnx.Module):
    """Pre-norm transformer block: LN -> Attention -> residual, LN -> MLP -> residual."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, *, rngs: nnx.Rngs):
        self.layernorm_before = nnx.LayerNorm(hidden_size, epsilon=1e-12, rngs=rngs)
        self.attention = ViTAttention(hidden_size, num_heads, rngs=rngs)
        self.layernorm_after = nnx.LayerNorm(hidden_size, epsilon=1e-12, rngs=rngs)
        self.mlp = ViTMLP(hidden_size, intermediate_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attention(self.layernorm_before(x))
        x = x + self.mlp(self.layernorm_after(x))
        return x


class ViTEncoder(nnx.Module):
    """Stack of ViT transformer blocks."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, num_layers: int, *, rngs: nnx.Rngs):
        self.layers = nnx.List([
            ViTBlock(hidden_size, num_heads, intermediate_size, rngs=rngs)
            for _ in range(num_layers)
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x


class TheiaEncoderJax(nnx.Module):
    """JAX/Flax NNX implementation of the Theia encoder (forward_feature path).

    Computes backbone features with CLS token stripped, returning [B, num_patches, hidden_size].
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        num_layers: int = 12,
        patch_size: int = 16,
        image_size: int = 224,
        num_channels: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        num_patches = (image_size // patch_size) ** 2
        self.embeddings = ViTEmbeddings(hidden_size, patch_size, num_patches, num_channels, rngs=rngs)
        self.encoder = ViTEncoder(hidden_size, num_heads, intermediate_size, num_layers, rngs=rngs)
        self.layernorm = nnx.LayerNorm(hidden_size, epsilon=1e-12, rngs=rngs)

    def __call__(self, pixel_values: jax.Array) -> jax.Array:
        """Forward pass returning backbone features with CLS token stripped.

        Args:
            pixel_values: preprocessed float32 images [B, C, H, W].

        Returns:
            Feature tensor [B, num_patches, hidden_size].
        """
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.layernorm(x)
        # Strip CLS token (index 0)
        return x[:, 1:]

    @classmethod
    def from_pretrained(cls, model_name: str) -> "TheiaEncoderJax":
        """Load a pretrained Theia model and convert weights to JAX.

        Args:
            model_name: HuggingFace model name (e.g., 'theaiinstitute/theia-base-patch16-224-cdiv').

        Returns:
            TheiaEncoderJax with pretrained weights loaded.
        """
        import torch
        from huggingface_hub import hf_hub_download

        from theia.models.jax.weight_conversion import convert_pytorch_to_jax

        # Download and load state_dict directly (avoids instantiating the full PyTorch model
        # which can fail with newer transformers versions)
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        except (ImportError, Exception):
            weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")

        # Infer config from state_dict shapes
        # backbone.model.embeddings.patch_embeddings.projection.weight: [hidden, channels, patch, patch]
        proj_weight = state_dict["backbone.model.embeddings.patch_embeddings.projection.weight"]
        hidden_size = proj_weight.shape[0]
        num_channels = proj_weight.shape[1]
        patch_size = proj_weight.shape[2]

        # backbone.model.embeddings.position_embeddings: [1, 1+num_patches, hidden]
        pos_embed = state_dict["backbone.model.embeddings.position_embeddings"]
        num_patches = pos_embed.shape[1] - 1
        image_size = int(num_patches ** 0.5) * patch_size

        # Count encoder layers by finding max layer index
        num_layers = 0
        for key in state_dict:
            if key.startswith("backbone.model.encoder.layer."):
                idx = int(key.split(".")[4])
                num_layers = max(num_layers, idx + 1)

        # Infer num_heads and intermediate_size from weight shapes
        # Q/K/V bias shape = [hidden_size], intermediate from MLP
        mlp_key = f"backbone.model.encoder.layer.0.intermediate.dense.weight"
        intermediate_size = state_dict[mlp_key].shape[0]

        # HuggingFace ViT uses hidden_size / num_attention_heads; infer from config
        # Standard DeiT: tiny=3, small=6, base=12 heads
        # We can infer from attention output dense weight or just use standard mapping
        head_size_to_num_heads = {192: 3, 384: 6, 768: 12}
        num_heads = head_size_to_num_heads.get(hidden_size, hidden_size // 64)

        # Create JAX model
        rngs = nnx.Rngs(0)
        jax_model = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            patch_size=patch_size,
            image_size=image_size,
            num_channels=num_channels,
            rngs=rngs,
        )

        # Convert and load weights
        convert_pytorch_to_jax(state_dict, jax_model)

        # Free state dict
        del state_dict

        return jax_model
