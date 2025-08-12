import cupy as cp
import time

# --- Parameters ---
# Adjust MATRIX_SIZE to increase or decrease the load.
# A larger size will put more stress on the GPU.
# For RTX 3080 Ti, 10000x10000 should provide a good load.
MATRIX_SIZE = 10000

# Number of repetitions for the matrix multiplication
NUM_REPETITIONS = 5

print(f"🚀 Starting GPU stress test with {NUM_REPETITIONS} repetitions of {MATRIX_SIZE}x{MATRIX_SIZE} matrix multiplication.")
print(f"   This will generate a significant load on your NVIDIA GeForce RTX 3080 Ti.")

try:
    # Set the default device to GPU 0 (your RTX 3080 Ti)
    cp.cuda.Device(0).use()

    # Get GPU memory info before starting
    mempool = cp.get_default_memory_pool()
    initial_used_gb = mempool.used_bytes() / (1024**3)
    initial_total_gb = mempool.total_bytes() / (1024**3)
    print(f"\n📊 Initial GPU Memory Used: {initial_used_gb:.2f} GB / {initial_total_gb:.2f} GB")

    total_time = 0

    for i in range(NUM_REPETITIONS):
        print(f"\nIteration {i+1}/{NUM_REPETITIONS}...")

        # Generate large random matrices directly on the GPU
        # Using float32 is generally faster and uses less memory
        start_gen = time.time()
        a_gpu = cp.random.rand(MATRIX_SIZE, MATRIX_SIZE, dtype=cp.float32)
        b_gpu = cp.random.rand(MATRIX_SIZE, MATRIX_SIZE, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize() # Ensure matrices are fully created
        end_gen = time.time()
        print(f"   Matrices generated on GPU in {end_gen - start_gen:.2f} seconds.")

        # Perform matrix multiplication on the GPU
        start_mul = time.time()
        c_gpu = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize() # Wait for GPU computation to complete
        end_mul = time.time()
        mul_time = end_mul - start_mul
        total_time += mul_time
        print(f"   Matrix multiplication completed on GPU in {mul_time:.2f} seconds.")

        # Optional: Print current GPU memory usage
        current_used_gb = mempool.used_bytes() / (1024**3)
        print(f"   Current GPU Memory Used: {current_used_gb:.2f} GB")

    avg_time = total_time / NUM_REPETITIONS
    print(f"\n✅ All {NUM_REPETITIONS} iterations completed!")
    print(f"Average multiplication time: {avg_time:.2f} seconds per iteration.")

    # Free up GPU memory
    mempool.free_all_blocks()
    final_used_gb = mempool.used_bytes() / (1024**3)
    print(f"✅ GPU Memory after freeing: {final_used_gb:.2f} GB")

except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"❌ CUDA Runtime Error: {e}")
    print("   This might indicate an issue with your GPU driver, CUDA Toolkit, or cuDNN installation.")
    print("   Ensure all previous steps (driver, CUDA, cuDNN, Visual Studio) were completed correctly.")
    print("   You might also try reducing MATRIX_SIZE if this is a memory issue.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    print("   Please ensure CuPy is installed (`pip install cupy`) and your environment variables are correctly set.")
