from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

n_points = int(2e7)

if rank == 0:
  my_n_points = n_points - (size - 1)*(n_points//size)
else:
  my_n_points = n_points//size

random.seed(rank)
my_sum = 0

for i in range(my_n_points):
  x = random.random()
  my_sum += 4 / (1 + x**2)

my_pi = np.array(my_sum / my_n_points, dtype='d')

pi = np.array(0, dtype='d')
comm.Reduce([my_pi, MPI.DOUBLE], [pi, MPI.DOUBLE], op=MPI.SUM, root=0)

pi /= size

if rank == 0:
  print('PI equals', pi)
