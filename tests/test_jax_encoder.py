# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoProcessor

from theia.models.jax import TheiaEncoderJax
from theia.models.rvfm import RobotVisionFM

MODEL_NAME = "theaiinstitute/theia-base-patch16-224-cdiv"
DEIT_NAME = "facebook/deit-base-patch16-224"


def _load_pytorch_model():
    """Load the PyTorch Theia model by constructing it and loading safetensors weights."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Build model structure with pretrained DeiT backbone (no translator needed)
    pt_model = RobotVisionFM(backbone=DEIT_NAME, pretrained=True)

    # Load Theia checkpoint weights on top
    weights_path = hf_hub_download(repo_id=MODEL_NAME, filename="model.safetensors")
    state_dict = load_file(weights_path)
    # Only load backbone weights
    backbone_keys = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
    pt_model.load_state_dict(backbone_keys, strict=False)
    pt_model.eval()
    return pt_model


@pytest.fixture(scope="module")
def models():
    """Load both PyTorch and JAX models once for all tests."""
    pt_model = _load_pytorch_model()
    jax_model = TheiaEncoderJax.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(DEIT_NAME)
    return pt_model, jax_model, processor


def _make_test_input(processor):
    """Create a deterministic preprocessed image for both frameworks."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Preprocess with HuggingFace processor
    inputs = processor(image, return_tensors="pt")
    pixel_values_pt = inputs["pixel_values"]  # [1, 3, 224, 224] float32
    pixel_values_jax = jnp.array(pixel_values_pt.numpy())
    return pixel_values_pt, pixel_values_jax


def test_numerical_equivalence(models):
    """Verify JAX and PyTorch models produce numerically equivalent outputs."""
    pt_model, jax_model, processor = models
    pixel_values_pt, pixel_values_jax = _make_test_input(processor)

    # PyTorch forward
    with torch.no_grad():
        pt_output = pt_model.forward_feature(
            pixel_values_pt,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
    pt_numpy = pt_output.numpy()

    # JAX forward
    jax_output = jax_model(pixel_values_jax)
    jax_numpy = np.array(jax_output)

    assert pt_numpy.shape == jax_numpy.shape, (
        f"Shape mismatch: PyTorch {pt_numpy.shape} vs JAX {jax_numpy.shape}"
    )

    max_diff = np.max(np.abs(pt_numpy - jax_numpy))
    mean_diff = np.mean(np.abs(pt_numpy - jax_numpy))

    print(f"\nMax abs diff: {max_diff:.2e}")
    print(f"Mean abs diff: {mean_diff:.2e}")
    print(f"Output shape: {pt_numpy.shape}")

    # Tolerances account for GPU vs CPU float32 differences across 12 transformer layers.
    # When both run on the same device (e.g., CPU), diffs are typically < 1e-5.
    assert max_diff < 5e-3, f"Max abs diff {max_diff:.2e} exceeds threshold 5e-3"
    assert mean_diff < 5e-4, f"Mean abs diff {mean_diff:.2e} exceeds threshold 5e-4"


def test_benchmark(models):
    """Benchmark both implementations and report timing."""
    pt_model, jax_model, processor = models
    pixel_values_pt, pixel_values_jax = _make_test_input(processor)
    num_iterations = 20

    # JAX warmup (JIT compilation)
    jax_fn = nnx.jit(jax_model)
    _ = jax_fn(pixel_values_jax)
    _ = jax_fn(pixel_values_jax)

    # JAX benchmark
    jax_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = jax_fn(pixel_values_jax)
        jax.block_until_ready(result)
        jax_times.append(time.perf_counter() - start)

    # PyTorch benchmark
    pt_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = pt_model.forward_feature(
                pixel_values_pt,
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
            )
            pt_times.append(time.perf_counter() - start)

    jax_mean = np.mean(jax_times) * 1000
    pt_mean = np.mean(pt_times) * 1000

    print(f"\nBenchmark ({num_iterations} iterations):")
    print(f"  JAX (JIT):     {jax_mean:.1f} ms/inference")
    print(f"  PyTorch (CPU): {pt_mean:.1f} ms/inference")
