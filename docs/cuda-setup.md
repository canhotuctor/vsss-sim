# CUDA setup

This guide covers JAX GPU use on Linux x86_64 with an NVIDIA GPU. The cuda extra is not available on macOS.

## Install

Use a current NVIDIA driver and verify that the device is visible:

    nvidia-smi

From the repository root, install the CUDA extra:

    python -m pip install -e '.[cuda]'

The project extra currently requests the JAX CUDA 12 pip build. Check the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) for current GPU and driver requirements.

## Select the GPU backend

Importing vsss_sim sets JAX_PLATFORMS to cpu unless it is already set, so simulator runs stay on the CPU by default. To use the GPU, export JAX_PLATFORMS=cuda before starting Python:

    export JAX_PLATFORMS=cuda

Setting it inside a running process has no effect once vsss_sim has been imported. Every vsss_sim entry point is affected, including scripts/check_gpu.py, scripts/train_jax.py, scripts/evaluate_jax.py, and scripts/bench_self_play.py. The benchmark scripts that accept --device gpu set the variable themselves. With --device default they now run on the CPU unless JAX_PLATFORMS is exported.

An experiment with "require_gpu": true stops before training when no GPU backend is active. Its error message shows the current JAX_PLATFORMS value. Without require_gpu, a run that forgets the variable trains on the CPU and prints CPU devices on its "JAX devices" line.

## Verify

Run the device diagnostic:

    JAX_PLATFORMS=cuda python scripts/check_gpu.py

The script reports the JAX backend, devices, and whether a small physics step runs on a GPU. Without JAX_PLATFORMS=cuda it reports only the CPU, even on a working GPU. If it reports only a CPU device with the variable set, check the driver installation and the JAX installation output.

## Measure performance

Measure physics and the batched Gymnasium path:

    python scripts/bench_backends.py --device gpu

Measure end-to-end self-play training and device memory:

    JAX_PLATFORMS=cuda python scripts/bench_self_play.py

JAX reserves 75% of GPU memory on its first operation, so nvidia-smi shows that reservation rather than what training uses. To measure real usage, add XLA_PYTHON_CLIENT_PREALLOCATE=false and read the peak_bytes_in_use field of the benchmark's device_memory output. The [performance and memory report](performance-memory-report.md) explains where training memory goes.

The benchmark output reports the active device. Record the hardware, JAX version, configuration, compilation time, and steady-state throughput with any results you keep.
