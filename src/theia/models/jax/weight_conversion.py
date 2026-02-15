# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def convert_pytorch_to_jax(state_dict: dict[str, Any], jax_model: Any) -> None:
    """Convert PyTorch state_dict weights and load them into a JAX TheiaEncoderJax model in-place.

    Handles the following transforms:
    - Conv weights: [out, in, kH, kW] -> [kH, kW, in, out]
    - Linear weights: [out, in] -> [in, out]
    - LayerNorm weight -> scale (direct copy)
    - Biases, cls_token, position_embeddings: direct copy

    Args:
        state_dict: PyTorch model state_dict (keys prefixed with 'backbone.model.').
        jax_model: TheiaEncoderJax instance to load weights into.
    """
    prefix = "backbone.model."

    def to_jax(tensor: Any) -> jnp.ndarray:
        return jnp.array(tensor.detach().cpu().float().numpy())

    # --- Embeddings ---
    # Patch embedding conv: [out, in, kH, kW] -> [kH, kW, in, out]
    conv_w = state_dict[f"{prefix}embeddings.patch_embeddings.projection.weight"]
    jax_model.embeddings.patch_embeddings.projection.kernel.value = jnp.array(
        np.transpose(conv_w.detach().cpu().float().numpy(), (2, 3, 1, 0))
    )
    jax_model.embeddings.patch_embeddings.projection.bias.value = to_jax(
        state_dict[f"{prefix}embeddings.patch_embeddings.projection.bias"]
    )

    # CLS token and position embeddings
    jax_model.embeddings.cls_token.value = to_jax(state_dict[f"{prefix}embeddings.cls_token"])
    jax_model.embeddings.position_embeddings.value = to_jax(state_dict[f"{prefix}embeddings.position_embeddings"])

    # --- Encoder layers ---
    num_layers = len(jax_model.encoder.layers)
    for i in range(num_layers):
        pt_prefix = f"{prefix}encoder.layer.{i}"
        jax_layer = jax_model.encoder.layers[i]

        # Attention Q/K/V + output
        for name, jax_linear in [
            ("query", jax_layer.attention.query),
            ("key", jax_layer.attention.key),
            ("value", jax_layer.attention.value),
        ]:
            w = state_dict[f"{pt_prefix}.attention.attention.{name}.weight"]
            b = state_dict[f"{pt_prefix}.attention.attention.{name}.bias"]
            jax_linear.kernel.value = to_jax(w).T
            jax_linear.bias.value = to_jax(b)

        # Attention output dense
        w = state_dict[f"{pt_prefix}.attention.output.dense.weight"]
        b = state_dict[f"{pt_prefix}.attention.output.dense.bias"]
        jax_layer.attention.output_dense.kernel.value = to_jax(w).T
        jax_layer.attention.output_dense.bias.value = to_jax(b)

        # LayerNorm before attention
        jax_layer.layernorm_before.scale.value = to_jax(state_dict[f"{pt_prefix}.layernorm_before.weight"])
        jax_layer.layernorm_before.bias.value = to_jax(state_dict[f"{pt_prefix}.layernorm_before.bias"])

        # LayerNorm after attention
        jax_layer.layernorm_after.scale.value = to_jax(state_dict[f"{pt_prefix}.layernorm_after.weight"])
        jax_layer.layernorm_after.bias.value = to_jax(state_dict[f"{pt_prefix}.layernorm_after.bias"])

        # MLP intermediate dense
        w = state_dict[f"{pt_prefix}.intermediate.dense.weight"]
        b = state_dict[f"{pt_prefix}.intermediate.dense.bias"]
        jax_layer.mlp.dense1.kernel.value = to_jax(w).T
        jax_layer.mlp.dense1.bias.value = to_jax(b)

        # MLP output dense
        w = state_dict[f"{pt_prefix}.output.dense.weight"]
        b = state_dict[f"{pt_prefix}.output.dense.bias"]
        jax_layer.mlp.dense2.kernel.value = to_jax(w).T
        jax_layer.mlp.dense2.bias.value = to_jax(b)

    # --- Final LayerNorm ---
    jax_model.layernorm.scale.value = to_jax(state_dict[f"{prefix}layernorm.weight"])
    jax_model.layernorm.bias.value = to_jax(state_dict[f"{prefix}layernorm.bias"])
