#!/bin/bash

#!/bin/bash

commands=(
    # Amazon
    "python test_run.py --experiment=stale --world_size=1 --partition_method=PSS --dataset=Amazon --timesteps=20 --model=TGCN --lr=0.001"
    # Epinion
    "python test_run.py --experiment=stale --world_size=1 --partition_method=PSS --dataset=Epinion --timesteps=20 --model=TGCN --lr=0.001"
    # Movie
    "python test_run.py --experiment=stale --world_size=1 --partition_method=PSS --dataset=Movie --timesteps=20 --model=TGCN --lr=0.001"
    # Stack
    "python test_run.py --experiment=stale --world_size=1 --partition_method=PSS --dataset=Stack --timesteps=20 --model=TGCN --lr=0.001"
)

for cmd in "${commands[@]}"
do
    $cmd
done