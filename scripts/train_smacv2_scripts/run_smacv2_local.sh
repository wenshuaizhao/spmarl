#!/bin/bash

env="SMACv2"
map="10gen_protoss"
teacher="spmarl"
algo="rmappo"
exp="spmarl4_local_check"

python ./train/train_smac.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed 100 --n_rollout_threads 10 --n_eval_rollout_threads 2 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 5 --use_eval --eval_episodes 32 --teacher ${teacher}
# --units ${units}