#!/bin/bash
INTERFACE="enp225s0"

MODEL=${1:-DySAT}
DATA=${2:-Amazon}
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache
PARTITION_STRATEGY="${6:-hash}"

HOST_NODE_ADDR=172.17.0.3
HOST_NODE_PORT=29400
NNODES=1
NPROC_PER_NODE=2

CURRENT_NODE_IP=$(ip -4 a show dev ${INTERFACE} | grep inet | cut -d " " -f6 | cut -d "/" -f1)
if [ $CURRENT_NODE_IP = $HOST_NODE_ADDR ]; then
    IS_HOST=true
else
    IS_HOST=false
fi

export NCCL_SOCKET_IFNAME=${INTERFACE}
export GLOO_SOCKET_IFNAME=${INTERFACE}
export TP_SOCKET_IFNAME=${INTERFACE}

cmd="torchrun \
    --standalone
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    test_run.py --model $MODEL --dataset $DATA"

echo $cmd
LOGLEVEL=INFO OMP_NUM_THREADS=2 exec $cmd


