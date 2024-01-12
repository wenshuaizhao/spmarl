#!/bin/sh

#SBATCH --job-name=mpe
#SBATCH --account=spmarl
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=../results/spmarl_output/job_%A_%a.out
#SBATCH --error=../results/spmarl_output/array_job_err_%A_%a.txt
#SBATCH --array=0-4

env="MPE"
scenario="simple_spread"
num_landmarks=8
num_agents=8
algo="rmappo" #"mappo" "ippo"
exp="check"
seed=1
perf_lb=55
reward_low=4
distance_thre=0.13

case $SLURM_ARRAY_TASK_ID in
   0)   teacher='spmarl';;
   1)   teacher='sprl';;
   2)   teacher='random';;
   3)   teacher='no_teacher';;
   4)   teacher='linear';;

esac

SLURM_CPUS_PER_TASK=42
srun --cpus-per-task=$SLURM_CPUS_PER_TASK singularity run \
--rocm -B $SCRATCH:$SCRATCH \
/scratch/spmarl/docker/mujo_gfoot_env_v2.sif \
/bin/sh -c \
"
export PYTHONUSERBASE=/scratch/spmarl/venv_pkgs/mujo_gfoot_env_v2; \
cd /scratch/spmarl/spmarl/scripts;\
pwd;\
python ./train/train_cmpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 256 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "spmarl_spmarl" \
    --user_name "spmarl_spmarl" --use_eval --use_partial_obs --obs_range 4 --n_eval_rollout_threads 100 \
    --teacher ${teacher} --perf_lb ${perf_lb} --sparse_reward --reward_low ${reward_low} --distance_thre ${distance_thre}"
