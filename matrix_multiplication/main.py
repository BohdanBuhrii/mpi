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

# rank of main node
MASTER = 0

# path to the folder with data
DATA_FOLDER = '/mnt/c/Buhrii_B/UnivAQ/Parallel Computing/mpi/matrix_multiplication/data/'

comm = MPI.COMM_WORLD
nprocs = MPI.COMM_WORLD.Get_size()  # get number of processes
rank = MPI.COMM_WORLD.Get_rank()    # get current process' id

# count: number of elements to send to each process
ave, res = divmod(N, nprocs)
result_count = np.array([ave + 1 if p < res else ave for p in range(nprocs)])
count = result_count * M

# displacement: the starting index of each process
result_displacement = np.array([sum(result_count[:p]) for p in range(nprocs)])
displacement = np.array([sum(count[:p]) for p in range(nprocs)])

if rank == MASTER:
  tic()
  # load initial data from the files
  matrix = np.load(DATA_FOLDER + '{}_{}.npy'.format(N, M), allow_pickle=True)
  vector = np.load(DATA_FOLDER + '{}.npy'.format(M), allow_pickle=True)
  toc('Data loaded')
else:
  matrix = None
  vector = np.empty((M,1), dtype='d')

# send vector to all processes
comm.Bcast(vector, root=0)

# initialize the submatrix for all processes
submatrix = np.empty(count[rank])

tic()
# distribute matrix between all processes
comm.Scatterv([matrix, count, displacement, MPI.DOUBLE], submatrix, root=MASTER)
toc('Scatterv')

tic()
# fix array dimensions
n = int(count[rank]/M)
submatrix = submatrix.reshape((n, M))
toc('Reshape')

tic()
# multiply the submatrix by the vector
partial_result = np.empty((n, 1))
for i in range(n):
  # iterate through the pairs of the elements, to avoid calling each element by index
  s = 0
  for a, b in zip(submatrix[i], vector):
    s += a * b
  partial_result[i] = s

toc('Multiply')

tic()
# gather data from all processes
result = np.empty((N, 1))
comm.Gatherv(partial_result, [result, result_count, result_displacement, MPI.DOUBLE], root=MASTER)
toc('Gather')

if rank == MASTER:
  tic()
  # save the result
  np.save(DATA_FOLDER + '{}_{}_result'.format(N, M), result)
  toc('Save')
