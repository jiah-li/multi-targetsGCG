#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=4
export model=$1 # llama2 or vicuna
export start=$2

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# for big_add in 0 20 40 60 80
for big_add in start
do
    for small_add in 0 4 8 12 16 
    do
        data_offset=$((big_add+small_add))
        python -u ../main.py \
            --config="../configs/transfer_${model}.py" \
            --config.attack=gcg \
            --config.train_data="../../data/advbench/harmful_behaviors.csv" \
            --config.result_prefix="../results/multiple_goals_4/transfer_${model}_gcg_${data_offset}_${n}_progressive" \
            --config.progressive_goals=False \
            --config.stop_on_success=True \
            --config.num_train_models=1 \
            --config.allow_non_ascii=False \
            --config.n_train_data=$n \
            --config.data_offset=$data_offset \
            --config.n_steps=500 \
            --config.test_steps=1 \
            --config.batch_size=512 \
            # --config.n_test_data=$n \
    done
done

# CUDA_VISIBLE_DEVICES=0 bash run_gcg_multiple.sh llama2 0
# CUDA_VISIBLE_DEVICES=4 bash run_gcg_multiple.sh llama2 20
# CUDA_VISIBLE_DEVICES=5 bash run_gcg_multiple.sh llama2 40
# CUDA_VISIBLE_DEVICES=6 bash run_gcg_multiple.sh llama2 60
# CUDA_VISIBLE_DEVICES=7 bash run_gcg_multiple.sh llama2 80