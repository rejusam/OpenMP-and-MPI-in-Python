from mpi4py import MPI
import numpy as np

# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate random data on each process
local_data = np.random.randint(0, 100, 100)

# Sum local data
local_sum = np.sum(local_data)

# Gather local sums onto root process
global_sums = comm.gather(local_sum, root=0)

# Calculate global sum on root process
if rank == 0:
    total_sum = np.sum(global_sums)
else:
    total_sum = None

# Broadcast total sum to all processes
total_sum = comm.bcast(total_sum, root=0)

# Calculate mean
mean = total_sum / (size * 100)

# Print mean on all processes
print(f"Process {rank}: Mean of data is {mean}")

