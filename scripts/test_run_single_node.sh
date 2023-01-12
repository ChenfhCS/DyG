#!/bin/bash

MODEL=${1:-DySAT}
DATA=${2:-Amazon}
NNODES=1
NPROC_PER_NODE=2

cmd="torchrun \
    --standalone
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    test_run.py --model $MODEL --dataset $DATA"

echo $cmd
LOGLEVEL=INFO OMP_NUM_THREADS=8 exec $cmd


