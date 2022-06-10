from mpi4py import MPI
import numpy as np
import time

N = 20000
M = 10000
MASTER = 0
# TODO hide this shit
DATA_FOLDER = '/mnt/c/Buhrii_B/UnivAQ/Parallel Computing/mpi/matrix_multiplication/data/'

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# consider moving count and displacement to master node only

# count: the size of each sub-task
ave, res = divmod(N, size)
result_count = np.array([ave + 1 if p < res else ave for p in range(size)])
count = result_count * M

# displacement: the starting index of each sub-task
result_displacement = np.array([sum(result_count[:p]) for p in range(size)])
displacement = np.array([sum(count[:p]) for p in range(size)])

if rank == MASTER:
  # get the data from somewhere
  matrix = np.load(DATA_FOLDER + '{}_{}.npy'.format(N, M), allow_pickle=True)
  vector = np.load(DATA_FOLDER + '{}.npy'.format(M), allow_pickle=True)
else:
  matrix = None
  vector = np.empty((M,1), dtype='d')

# send vector to all nodes
comm.Bcast(vector, root=0)

# initialize the submatrix for all processes
# try to do something with the shape
submatrix = np.empty(count[rank])

# distribute data between all processes
comm.Scatterv([matrix, count, displacement, MPI.DOUBLE], submatrix, root=MASTER)
# fix dimensions
submatrix = submatrix.reshape((int(count[rank]/M), M))

#print('Process {} has data:\n'.format(rank), submatrix)

# compute partial result
partial_result = np.dot(submatrix, vector)

# gather data from all processes
result = np.empty((N, 1))
comm.Gatherv(partial_result, [result, result_count, result_displacement, MPI.DOUBLE], root=MASTER)

# show the result
if rank == MASTER:
  np.save(DATA_FOLDER + '{}_{}_result'.format(N, M), result)
  print('DONE!!!')

