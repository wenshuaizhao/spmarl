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
#SBATCH --array=0-19

env="SMACv2"
map="10gen_protoss"
# teacher=$1
units="15v5" #"15v9"
target_mean=5

algo="rmappo"
exp="icml2025"
perf_lb=0.6

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

case $SLURM_ARRAY_TASK_ID in
0) teacher='invlinear' ;;
1) teacher='alpgmm' ;;
2) teacher='vacl' ;;
3) teacher='no_teacher' ;;
4) teacher='invlinear' ;;
5) teacher='alpgmm' ;;
6) teacher='vacl' ;;
7) teacher='no_teacher' ;;
8) teacher='invlinear' ;;
9) teacher='alpgmm' ;;
10) teacher='vacl' ;;
11) teacher='no_teacher' ;;
12) teacher='invlinear' ;;
13) teacher='alpgmm' ;;
14) teacher='vacl' ;;
15) teacher='no_teacher' ;;
16) teacher='invlinear' ;;
17) teacher='alpgmm' ;;
18) teacher='vacl' ;;
19) teacher='no_teacher' ;;

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
