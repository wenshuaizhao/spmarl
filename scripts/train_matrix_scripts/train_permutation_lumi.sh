#!/bin/bash
#SBATCH --job-name=matrix-rmappo
#SBATCH --output=./out/matrix-rmappo_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/matrix-rmappo_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=small
#SBATCH --account=project_id
#SBATCH --array=0-4

env="matrix"
scenario="permutation"

algo="rmappo"
exp="check"
teacher=$1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}"

srun singularity exec -B $SCRATCH $SCRATCH/spmarl.sif python ../train/train_matrix.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed $SLURM_ARRAY_TASK_ID --n_rollout_threads 50 --n_eval_rollout_threads 2 --num_mini_batch 1 --episode_length 200 \
--num_env_steps 10000000 --ppo_epoch 5 --use_eval --eval_episodes 32 --teacher ${teacher}
# --units ${units} 
