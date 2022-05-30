from mpi4py import MPI
import numpy as np
import random

comm = MPI.Comm.Get_parent()
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
print(rank)

n_points = np.array(0, dtype='i')
comm.Bcast([n_points, MPI.INT], root=0)
# n_points is an array, not an integer
if rank == 0:
  my_n_points = n_points - (size - 1)*(n_points//size)
else:
  my_n_points = n_points//size

random.seed(rank)
my_sum = 0

for i in range(my_n_points):
  x = random.random()
  my_sum += 4 / (1 + x**2)

pi = np.array(my_sum / my_n_points, dtype='d')
comm.Reduce([pi, MPI.DOUBLE], None,
            op=MPI.SUM, root=0)

comm.Disconnect()
