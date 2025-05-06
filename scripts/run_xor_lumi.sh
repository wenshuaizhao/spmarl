#!/bin/sh

#SBATCH --job-name=xor
#SBATCH --account=project_id
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=../results/lumi_output/job_%A_%a.out
#SBATCH --error=../results/lumi_output/array_job_err_%A_%a.txt
#SBATCH --array=0-25

env="matrix"
scenario="permutation"
exp="icml2025"
algo="rmappo"

case $SLURM_ARRAY_TASK_ID in
   0)   teacher='spmarl';;
   1)   teacher='sprl';;
   2)   teacher='spmarl';;
   3)   teacher='sprl';;
   4)   teacher='spmarl';;
   5)   teacher='sprl';;
   6)   teacher='spmarl';;
   7)   teacher='sprl';;
   8)   teacher='spmarl';;
   9)   teacher='sprl';;
   10)  teacher='vacl';;
   11)  teacher='linear';;
   12)  teacher='linear';;
   13)  teacher='linear';;
   14)  teacher='linear';;
   15)  teacher='alpgmm';;
   16)  teacher='alpgmm';;
   17)  teacher='alpgmm';;
   18)  teacher='alpgmm';;
   19)  teacher='alpgmm';;
   20)  teacher='vacl';;
   21)  teacher='vacl';;
   22)  teacher='vacl';;
   23)  teacher='vacl';;
   24)  teacher='vacl';;
   25)  teacher='linear';;

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
python ./train/train_matrix.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--scenario_name ${scenario} --n_rollout_threads 50 --n_eval_rollout_threads 2 --num_mini_batch 1 --episode_length 200 \
--num_env_steps 10000000 --ppo_epoch 5 --use_eval --eval_episodes 32 --teacher ${teacher}"