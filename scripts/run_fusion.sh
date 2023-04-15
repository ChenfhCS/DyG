#!/bin/bash

#!/bin/bash

commands=(
    "python test_run.py --experiment=fusion --dataset=Amazon --timesteps=10 --world_size=1 --model=GC-LSTM --partition_method=PGC"
    "python test_run.py --experiment=fusion --dataset=Epinion --timesteps=30 --world_size=1 --model=GC-LSTM --partition_method=PGC"
    "python test_run.py --experiment=fusion --dataset=Movie --timesteps=30 --world_size=1 --model=GC-LSTM --partition_method=PGC"
    "python test_run.py --experiment=fusion --dataset=Stack --timesteps=30 --world_size=1 --model=GC-LSTM --partition_method=PGC"
)

for cmd in "${commands[@]}"
do
    $cmd
done