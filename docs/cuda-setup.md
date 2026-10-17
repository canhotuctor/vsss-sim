# CUDA setup

This guide covers JAX GPU use on Linux x86_64 with an NVIDIA GPU. The cuda extra is not available on macOS.

## Install

Use a current NVIDIA driver and verify that the device is visible:

    nvidia-smi

From the repository root, install the CUDA extra:

    python -m pip install -e '.[cuda]'

The project extra currently requests the JAX CUDA 12 pip build. Check the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) for current GPU and driver requirements.

## Verify

Run the device diagnostic:

    python scripts/check_gpu.py

The script reports the JAX backend, devices, and whether a small physics step runs on a GPU. If it reports only a CPU device, check the driver installation, JAX installation output, and JAX-related environment variables such as JAX_PLATFORMS.

## Measure performance

Measure physics and the batched Gymnasium path:

    python scripts/bench_backends.py --device gpu

Measure end-to-end self-play training and device memory:

    python scripts/bench_self_play.py

The benchmark output reports the active device. Record the hardware, JAX version, configuration, compilation time, and steady-state throughput with any results you keep.
