# GPU Setup (Optional)

The analysis runs on CPU in under ~10 minutes. GPU is optional.

## Windows (NVIDIA CUDA)
- Install a recent NVIDIA driver (CUDA 12.x capable).
- (Optional) Install CUDA Toolkit 12.1 or 12.3 from NVIDIA.
- Python packages:
  ```
  pip install cupy-cuda12x numba
  ```
- Verify:
  ```
  python -c "import cupy; import cupy.cuda.runtime as r; print(r.getDeviceProperties(0)['name'])"
  ```
- If `cupy` fails to import, ensure your driver supports CUDA 12 and that you used the `cupy-cuda12x` build.

## macOS (Apple Silicon / Intel)
- CUDA is not available. Use CPU (default).
- Optional: install `numba` for CPU-optimized code paths.
  ```
  pip install numba
  ```

## Linux (NVIDIA)
- Similar to Windows: install NVIDIA driver + CUDA 12.x; then `pip install cupy-cuda12x`.

The scripts automatically fall back to NumPy if CuPy/Numba are unavailable.