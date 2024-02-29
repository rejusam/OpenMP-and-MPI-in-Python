import omp
import numpy as np

def parallel_computation():
    result = 0
    # Number of threads picked up from environment variable
    with omp.Parallel() as p:
        # Print thread number    
        print(f'Thread {p.thread_num} out of {p.num_threads} threads')
        # Each thread works on partial input
        chunk_size = int(1e6 / p.num_threads)
        result += np.sum(np.random.rand(chunk_size))  
    return result

print(parallel_computation())
