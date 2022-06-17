#!/bin/bash

# Output file
interpreter="/home/guestpar11/anaconda3/bin/python3.7"
output="/home/guestpar11/project/matrix_multiplication/output"
filename="/home/guestpar11/project/matrix_multiplication/main.py"

# List with Number of processes to try
nproc=(1 2 4 8 16 32 64 80)

# Size of the matrix
N=7000
M=7000

echo "N: $N; M: $M; nproc: ${nproc[@]}" >> $output
echo "N: $N; M: $M"
for proc in ${nproc[@]};
do
    echo "------------------------------------------------"
    echo "nproc: $proc"
    # Running program and saving the time to the output file
    { /usr/bin/time -f "%e" mpiexec -n $proc --hostfile hostfile_80 $interpreter $filename $N $M; } 2>> $output
done