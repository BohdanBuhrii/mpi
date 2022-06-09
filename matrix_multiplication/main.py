from re import sub
from mpi4py import MPI
import numpy as np

N = 7
M = 5
MASTER = 0

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# count: the size of each sub-task
ave, res = divmod(N, size)
result_count = np.array([ave + 1 if p < res else ave for p in range(size)])
count = result_count * M

# displacement: the starting index of each sub-task
result_displacement = np.array([sum(result_count[:p]) for p in range(size)])
displacement = np.array([sum(count[:p]) for p in range(size)])

# TODO send it to other nodes
# vector = np.random.rand(M, 1)
vector = np.arange(1, M + 1)

if rank == MASTER:
  # get the data from somewhere
  matrix = np.eye(N, M)
  #print(matrix)
else:
  matrix = None

# initialize the submatrix for all processes
submatrix = np.empty(count[rank])

# distribute data between all processes
comm.Scatterv([matrix, count, displacement, MPI.DOUBLE], submatrix, root=0)
# fix dimensions
submatrix = submatrix.reshape((int(count[rank]/M), M))

print('Process {} has data:\n'.format(rank), submatrix)

# compute partial result
partial_result = np.dot(submatrix, vector)

# gather data from all processes
result = np.empty((N, 1))
comm.Gatherv(partial_result, [result, result_count, result_displacement, MPI.DOUBLE], root=0)

# show the result
if rank == 0:
  print('Result:')
  print(result.round(2))
