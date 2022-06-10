from mpi4py import MPI
import numpy as np
import time

def tic():
  global tic_time
  tic_time = time.time()

def toc(message: str):
  print(rank, message, ' '*(12 - len(message)), np.round(time.time() - tic_time, 3))

N = 5000
M = 5000
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
  tic()
  matrix = np.load(DATA_FOLDER + '{}_{}.npy'.format(N, M), allow_pickle=True)
  vector = np.load(DATA_FOLDER + '{}.npy'.format(M), allow_pickle=True)
  toc('Data loaded')
else:
  matrix = None
  vector = np.empty((M,1), dtype='d')

# send vector to all nodes
comm.Bcast(vector, root=0)

# initialize the submatrix for all processes
# try to do something with the shape
submatrix = np.empty(count[rank])

# distribute data between all processes
tic()
comm.Scatterv([matrix, count, displacement, MPI.DOUBLE], submatrix, root=MASTER)
toc('Scatterv')
# fix dimensions
tic()
n = int(count[rank]/M)
submatrix = submatrix.reshape((n, M))
toc('Reshape')
#print('Process {} has data:\n'.format(rank), submatrix)

# compute partial result
# tic()
# this method is too good, I have to replace it with some shitty code
# partial_result = np.dot(submatrix, vector)
# toc('Dot')

tic()
# too slow
# partial_result = np.zeros((n, 1))
# for i in range(n):
#   for j in range(M):
#     partial_result[i] += submatrix[i][j]*vector[j]


partial_result = np.empty((n, 1))
for i in range(n):
  s = 0
  for a, b in zip(submatrix[i], vector):
    s += a * b
  
  partial_result[i] = s

toc('Multiply')

# gather data from all processes
tic()
result = np.empty((N, 1))
comm.Gatherv(partial_result, [result, result_count, result_displacement, MPI.DOUBLE], root=MASTER)
toc('Gather')

# show the result
if rank == MASTER:
  tic()
  np.save(DATA_FOLDER + '{}_{}_result'.format(N, M), result)
  toc('Save')
  print('DONE!!!')

