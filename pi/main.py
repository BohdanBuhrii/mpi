from mpi4py import MPI
import numpy as np
import sys

n_procs = 4
n_points = 2e5

#size = MPI.COMM_WORLD.Get_size()
#rank = MPI.COMM_WORLD.Get_rank()
#name = MPI.Get_processor_name()

comm = MPI.COMM_SELF.Spawn(sys.executable, args=['child.py'], maxprocs=n_procs)

n = np.array(n_points, 'i')
comm.Bcast([n, MPI.INT], root=MPI.ROOT)
pi = np.array(0.0, 'd')
comm.Reduce(None, [pi, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
print(pi)

comm.Disconnect()