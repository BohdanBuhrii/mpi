from mpi4py import MPI
import numpy as np

N = 10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# count: the size of each sub-task
ave, res = divmod(N, size)
count = np.array([ave + 1 if p < res else ave for p in range(size)])

# displacement: the starting index of each sub-task
displacement = np.array([sum(count[:p]) for p in range(size)])

if rank == 0:
    print('Count:', count)
    print('Displacement', displacement)

    # generate some data
    all_data = np.ones(N)
else:
    all_data = None

# initialize piece_of_data on all processes
piece_of_data = np.empty(count[rank])

# distribute data between all processes
comm.Scatterv([all_data, count, displacement, MPI.DOUBLE], piece_of_data, root=0)
print('Process {} has data:'.format(rank), piece_of_data)

# perform some computations
piece_of_data *= rank

# gather data from all processes
comm.Gatherv(piece_of_data, [all_data, count, displacement, MPI.DOUBLE], root=0)

# show the result
if rank == 0:
    print('Process 0 has data:', all_data)