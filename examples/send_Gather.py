from mpi4py import MPI
import numpy as np

MASTER = 0

# info about processes
comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# send buffer
sbuff = np.empty(2, dtype='i')
# receive buffer
rbuff = np.empty((size, 2),dtype='i')

if rank == MASTER:
  print('Master here')
  sbuff = np.zeros(2, dtype='i') + 10

  for i in range(1, size):
    # sending some random data to other nodes
    comm.Send([np.array([0, i], dtype='i'), MPI.INT], dest=i, tag=11)
    print('Sent to node', i)

else:
  
  comm.Recv([sbuff, MPI.INT], source=MASTER, tag=11)
  print('Received!!!', sbuff)
  print('Sending back ...')
  

comm.Gather(sbuff, rbuff, root=MASTER)

if rank == MASTER:
  print('All done!')
  print(rbuff.reshape((1, size*2)))

  
  