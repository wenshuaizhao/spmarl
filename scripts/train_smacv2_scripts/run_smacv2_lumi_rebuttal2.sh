#!/bin/sh

#SBATCH --job-name=smacv2_spmarl
#SBATCH --account=project_id
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=../results/lumi_output/job_%A_%a.out
#SBATCH --error=../results/lumi_output/array_job_err_%A_%a.txt
#SBATCH --array=0-4

env="SMACv2"
# map="10gen_protoss"
# map="10gen_terran"
map="10gen_zerg"
# teacher=$1
units="15v5" #"15v9"
target_mean=5

algo="rmappo"
exp="icml2025"
perf_lb=0.5

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

case $SLURM_ARRAY_TASK_ID in
0) teacher='invlinear' ;;
1) teacher='invlinear' ;;
2) teacher='invlinear' ;;
3) teacher='invlinear' ;;
4) teacher='invlinear' ;;
5) teacher='alpgmm' ;;
6) teacher='alpgmm' ;;
7) teacher='alpgmm' ;;
8) teacher='alpgmm' ;;
9) teacher='vacl' ;;
10) teacher='vacl' ;;
11) teacher='vacl' ;;
12) teacher='alpgmm' ;;
13) teacher='vacl' ;;
14) teacher='vacl' ;;
15) teacher='vacl' ;;
16) teacher='vacl' ;;
17) teacher='alpgmm' ;;
18) teacher='vacl' ;;
19) teacher='spmarl' ;;
20) teacher='invlinear' ;;
21) teacher='alpgmm' ;;
22) teacher='vacl' ;;
23) teacher='spmarl' ;;
24) teacher='sprl' ;;
25) teacher='sprl' ;;
26) teacher='sprl' ;;
27) teacher='sprl' ;;
28) teacher='sprl' ;;
29) teacher='sprl' ;;

esac

# case $SLURM_ARRAY_TASK_ID in
# 0) teacher='invlinear' ;;
# 1) teacher='alpgmm' ;;
# 2) teacher='vacl' ;;
# 3) teacher='spmarl' ;;
# 4) teacher='sprl' ;;
# 5) teacher='invlinear' ;;
# 6) teacher='alpgmm' ;;
# 7) teacher='vacl' ;;
# 8) teacher='spmarl' ;;
# 9) teacher='sprl' ;;
# 10) teacher='invlinear' ;;
# 11) teacher='alpgmm' ;;
# 12) teacher='vacl' ;;
# 13) teacher='spmarl' ;;
# 14) teacher='sprl' ;;
# 15) teacher='invlinear' ;;
# 16) teacher='alpgmm' ;;
# 17) teacher='vacl' ;;
# 18) teacher='spmarl' ;;
# 19) teacher='sprl' ;;
# 20) teacher='invlinear' ;;
# 21) teacher='alpgmm' ;;
# 22) teacher='vacl' ;;
# 23) teacher='spmarl' ;;
# 24) teacher='sprl' ;;

# esac

SLURM_CPUS_PER_TASK=16
srun --cpus-per-task=$SLURM_CPUS_PER_TASK singularity run \
	--rocm -B $SCRATCH:$SCRATCH \
	/scratch/project_id/docker/mujo_gfoot_env_v2.sif \
	/bin/sh -c \
	"
export SC2PATH=/users/user_name/StarCraftII; \
export SC2_RENDERER=sw; \
export PYTHONUSERBASE=/scratch/project_id/docker/venvs/spmarl4; \
python ./train/train_smac.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed $SLURM_ARRAY_TASK_ID --n_rollout_threads 25 --n_eval_rollout_threads 16 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 5 --use_eval --eval_episodes 64 --teacher ${teacher} --target_mean ${target_mean} --units ${units} --perf_lb ${perf_lb}
"

# --units ${units}
