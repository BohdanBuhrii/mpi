# mpi
Parallel matrix-vector multiplication 

##
To install mpi:
```
sudo apt-get update
sudo apt-get -y install mpich
```

sudo apt-get -y install openmpi-bin

##
Run the script with 4 processors:
```
mpiexec -n 4 python3 main.py
```

## Useful links
https://mpi4py.readthedocs.io/en/stable/intro.html
https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/