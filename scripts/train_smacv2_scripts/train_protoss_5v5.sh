#!/bin/bash

#SBATCH --job-name=smac-spmarl
#SBATCH --output=./out/smac-spmarl_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/smac-spmarl_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --array=0-4

#--gres=gpu:1

echo "export SC2PATH=/scratch/work/liz23/3rdparty/StarCraftII/" >> ~/.bashrc

env="SMACv2"
map="10gen_protoss"
teacher=$1
# units="5v5"

algo="rmappo"
exp="check"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../train/train_smac.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed $SLURM_ARRAY_TASK_ID --n_rollout_threads 25 --n_eval_rollout_threads 2 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 5 --use_eval --eval_episodes 32 --teacher ${teacher}
# --units ${units} 
