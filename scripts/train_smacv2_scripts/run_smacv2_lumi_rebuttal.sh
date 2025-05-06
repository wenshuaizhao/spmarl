#!/bin/sh

#SBATCH --job-name=smacv2_spmarl
#SBATCH --account=project_id
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=../results/lumi_output/job_%A_%a.out
#SBATCH --error=../results/lumi_output/array_job_err_%A_%a.txt
#SBATCH --array=0-1

env="SMACv2"
map="10gen_protoss"
# teacher=$1
units="15v5" #"15v9"
target_mean=5

algo="rmappo"
exp="icml2025"
# perf_lb=0.5
teacher='spmarl'

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

case $SLURM_ARRAY_TASK_ID in
0) perf_lb=0.4 ;;
1) perf_lb=0.4 ;;
2) perf_lb=0.7 ;;
3) perf_lb=0.7 ;;
4) perf_lb=0.8 ;;
5) perf_lb=0.8 ;;
6) perf_lb=0.5 ;;
7) perf_lb=0.4;;
8) perf_lb=0.4 ;;
9) perf_lb=0.4 ;;
10) perf_lb=0.4 ;;
11) perf_lb=0.4 ;;
12) perf_lb=0.4 ;;
13) perf_lb=0.4 ;;
14) perf_lb=0.3;;
15) perf_lb=0.3 ;;
16) perf_lb=0.3 ;;
17) perf_lb=0.3 ;;
18) perf_lb=0.3 ;;
19) perf_lb=0.3 ;;
20) perf_lb=0.3 ;;

esac

SLURM_CPUS_PER_TASK=16
srun --cpus-per-task=$SLURM_CPUS_PER_TASK singularity run \
	--rocm -B $SCRATCH:$SCRATCH \
	/scratch/project_id/docker/mujo_gfoot_env_v2.sif \
	/bin/sh -c \
	"
export SC2PATH=/users/user_name/StarCraftII; \
export PYTHONUSERBASE=/scratch/project_id/docker/venvs/spmarl4; \
python ./train/train_smac.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed $SLURM_ARRAY_TASK_ID --n_rollout_threads 25 --n_eval_rollout_threads 16 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 5 --use_eval --eval_episodes 64 --teacher ${teacher} --target_mean ${target_mean} --units ${units} --perf_lb ${perf_lb}
"

# --units ${units}
