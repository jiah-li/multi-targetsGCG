#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=4
export model=$1 # llama2 or vicuna

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# for big_add in 0 20 40 60 80
for big_add in 0
do
    for small_add in 0 4 8 12 16 
    do
        data_offset=$((big_add+small_add))
        python -u ../main.py \
            --config="../configs/transfer_${model}.py" \
            --config.attack=gcg \
            --config.train_data="../../data/advbench/harmful_behaviors.csv" \
            --config.result_prefix="../results/multiple_goals_4/transfer_${model}_gcg_${n}_progressive" \
            --config.progressive_goals=False \
            --config.stop_on_success=True \
            --config.num_train_models=1 \
            --config.allow_non_ascii=False \
            --config.n_train_data=$n \
            # --config.n_test_data=$n \
            --config.data_offset=$data_offset \
            --config.n_steps=1000 \
            --config.test_steps=1 \
            --config.batch_size=512
    done
done