#!/bin/bash

#!/bin/bash

commands=(
    # fig
    "python test_run.py --timesteps=10 --dataset=Amazon --model=TGCN --partition_method=PTS"
    "python test_run.py --timesteps=30 --dataset=Amazon --model=TGCN --partition_method=PSS"
    "python test_run.py --timesteps=30 --dataset=Amazon --model=TGCN --partition_method=PGC"
    "python test_run.py --dataset=Epinion --model=TGCN --partition_method=PTS"
    "python test_run.py --dataset=Epinion --model=TGCN --partition_method=PSS"
    "python test_run.py --dataset=Epinion --model=TGCN --partition_method=PGC"
    "python test_run.py --dataset=Movie --model=TGCN --partition_method=PTS"
    "python test_run.py --dataset=Movie --model=TGCN --partition_method=PSS"
    "python test_run.py --dataset=Movie --model=TGCN --partition_method=PGC"
    "python test_run.py --dataset=Stack --model=TGCN --partition_method=PTS"
    "python test_run.py --dataset=Stack --model=TGCN --partition_method=PSS"
    "python test_run.py --dataset=Stack --model=TGCN --partition_method=PGC"
    # fig 2
    "python test_run.py --timesteps=30 --dataset=Amazon --model=GC-LSTM --partition_method=PTS"
    "python test_run.py --timesteps=30 --dataset=Amazon --model=GC-LSTM --partition_method=PSS"
    "python test_run.py --timesteps=30 --dataset=Amazon --model=GC-LSTM --partition_method=PGC"
    "python test_run.py --dataset=Epinion --model=GC-LSTM --partition_method=PTS"
    "python test_run.py --dataset=Epinion --model=GC-LSTM --partition_method=PSS"
    "python test_run.py --dataset=Epinion --model=GC-LSTM --partition_method=PGC"
    "python test_run.py --dataset=Movie --model=GC-LSTM --partition_method=PTS"
    "python test_run.py --dataset=Movie --model=GC-LSTM --partition_method=PSS"
    "python test_run.py --dataset=Movie --model=GC-LSTM --partition_method=PGC"
    "python test_run.py --dataset=Stack --model=GC-LSTM --partition_method=PTS"
    "python test_run.py --dataset=Stack --model=GC-LSTM --partition_method=PSS"
    "python test_run.py --dataset=Stack --model=GC-LSTM --partition_method=PGC"
    # fig 3
    "python test_run.py --timesteps=30 --dataset=Amazon --model=MPNN-LSTM --partition_method=PTS"
    "python test_run.py --timesteps=30 --dataset=Amazon --model=MPNN-LSTM --partition_method=PSS"
    "python test_run.py --timesteps=30 --dataset=Amazon --model=MPNN-LSTM --partition_method=PGC"
    "python test_run.py --dataset=Epinion --model=MPNN-LSTM --partition_method=PTS"
    "python test_run.py --dataset=Epinion --model=MPNN-LSTM --partition_method=PSS"
    "python test_run.py --dataset=Epinion --model=MPNN-LSTM --partition_method=PGC"
    "python test_run.py --dataset=Movie --model=MPNN-LSTM --partition_method=PTS"
    "python test_run.py --dataset=Movie --model=MPNN-LSTM --partition_method=PSS"
    "python test_run.py --dataset=Movie --model=MPNN-LSTM --partition_method=PGC"
    "python test_run.py --dataset=Stack --model=MPNN-LSTM --partition_method=PTS"
    "python test_run.py --dataset=Stack --model=MPNN-LSTM --partition_method=PSS"
    "python test_run.py --dataset=Stack --model=MPNN-LSTM --partition_method=PGC"
    
)

for cmd in "${commands[@]}"
do
    $cmd
done