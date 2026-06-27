import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(managed_memory=True, pool_allocator=True, initial_pool_size="40GB")
import cupy as cp

cp.cuda.set_allocator(rmm_cupy_allocator)
import runpy
import sys

sys.argv = ["gpu_tpch_driver.py", *sys.argv[1:]]
runpy.run_path("/home/ubuntu/gpu_tpch_driver.py", run_name="__main__")
