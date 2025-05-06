#!/bin/sh

#SBATCH --job-name=mpe
#SBATCH --account=project_id
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../results/lumi_output/job_%A_%a.out
#SBATCH --error=../results/lumi_output/array_job_err_%A_%a.txt
#SBATCH --array=0-11

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

case $SLURM_ARRAY_TASK_ID in
   0)   teacher='alpgmm';;
   1)   teacher='vacl';;
   2)   teacher='alpgmm';;
   3)   teacher='vacl';;
   4)   teacher='alpgmm';;
   5)   teacher='vacl';;
   6)   teacher='alpgmm';;
   7)   teacher='vacl';;
   8)   teacher='alpgmm';;
   9)   teacher='vacl';;
   10)  teacher='alpgmm';;
   11)  teacher='vacl';;

esac

SLURM_CPUS_PER_TASK=42
srun --cpus-per-task=$SLURM_CPUS_PER_TASK singularity run \
--rocm -B $SCRATCH:$SCRATCH \
/scratch/project_id/docker/mujo_gfoot_env_v2.sif \
/bin/sh -c \
"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/user_name/.mujoco/mujoco210/bin; \
export PYTHONUSERBASE=/scratch/project_id/docker/venvs/spmarl4; \
cd /scratch/project_id/projects/spmarl4/scripts;\
pwd;\
python ./train/train_cmpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 256 --num_mini_batch 1 --episode_length 25 --num_env_steps 10000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "wszhao_aalto" \
    --user_name "wszhao_aalto" --use_eval --use_partial_obs --obs_range 4 --n_eval_rollout_threads 100 \
    --teacher ${teacher} --perf_lb ${perf_lb} --sparse_reward --reward_low ${reward_low} --distance_thre ${distance_thre}"