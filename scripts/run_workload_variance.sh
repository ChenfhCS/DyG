#!/bin/bash

#!/bin/bash

# 定义命令列表
commands=(
    # Amazon
    "python test_run.py --dataset=Amazon --world_size=2 --partition_method=PGC --model=MPNN-LSTM"
    # Epinion
    "python test_run.py --dataset=Epinion --world_size=2 --partition_method=PGC --model=MPNN-LSTM"
    # Movie
    "python test_run.py --dataset=Movie --world_size=2 --partition_method=PGC --model=MPNN-LSTM"
    # Stack
    "python test_run.py --dataset=Stack --world_size=2 --partition_method=PGC --model=MPNN-LSTM"
)

# 循环遍历命令列表，依次执行命令
for cmd in "${commands[@]}"
do
    $cmd
done