#!/bin/bash

# add mpi into PATH

# default arguments
nb_gpus=2

extra_args=''

echo "multi-GPU training enabled"
mpirun -np ${nb_gpus} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python top/train.py --enbl_multi_gpu  

