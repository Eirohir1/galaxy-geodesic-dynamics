import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

N = 10**7
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.empty_like(a)

mod = SourceModule("""
__global__ void add_vec(float *a, float *b, float *c, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}
""")

func = mod.get_function("add_vec")
threads_per_block = 1024
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

import time
start = time.time()

func(
    drv.In(a), drv.In(b), drv.Out(c), np.int32(N),
    block=(threads_per_block, 1, 1),
    grid=(blocks_per_grid, 1)
)

drv.Context.synchronize()
print("Kernel execution time:", time.time() - start, "seconds")
