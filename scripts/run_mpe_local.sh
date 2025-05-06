#!/bin/sh


env="MPE"
scenario="simple_spread"
num_landmarks=8
num_agents=8
algo="rmappo" #"mappo" "ippo"
exp="check"
seed=1
perf_lb=50
reward_low=4
distance_thre=0.13
teacher="spmarl"
cd ./scripts;

python ./train/train_cmpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 256 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "wszhao_aalto" \
    --user_name "wszhao_aalto" --use_eval --use_partial_obs --obs_range 4 --n_eval_rollout_threads 100 \
    --teacher ${teacher} --perf_lb ${perf_lb} --sparse_reward --reward_low ${reward_low} --distance_thre ${distance_thre}