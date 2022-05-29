# mpi
Parallel matrix-vector multiplication 

##
To install mpi:
```
sudo apt-get update
sudo apt-get -y install mpich
```

##
run the script with 4 processors
```
mpiexec -n 4 python main.py
```