#!/bin/sh

#SBATCH --job-name=smacv2_spmarl
#SBATCH --account=project_id
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=../results/lumi_output/job_%A_%a.out
#SBATCH --error=../results/lumi_output/array_job_err_%A_%a.txt
#SBATCH --array=0-9

env="VMAS"
scenario_name="reverse_transport"
target_mean=6
num_env_steps=2000000
algo="rmappo"
exp="icml2025"
perf_lb=35

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"


case $SLURM_ARRAY_TASK_ID in
0) teacher='linear' ;;
1) teacher='alpgmm' ;;
2) teacher='vacl' ;;
3) teacher='spmarl' ;;
4) teacher='sprl' ;;
5) teacher='linear' ;;
6) teacher='alpgmm' ;;
7) teacher='vacl' ;;
8) teacher='spmarl' ;;
9) teacher='sprl' ;;
10) teacher='linear' ;;
11) teacher='alpgmm' ;;
12) teacher='vacl' ;;
13) teacher='spmarl' ;;
14) teacher='sprl' ;;
15) teacher='linear' ;;
16) teacher='alpgmm' ;;
17) teacher='vacl' ;;
18) teacher='spmarl' ;;
19) teacher='sprl' ;;
20) teacher='linear' ;;
21) teacher='alpgmm' ;;
22) teacher='vacl' ;;
23) teacher='spmarl' ;;
24) teacher='sprl' ;;

esac

SLURM_CPUS_PER_TASK=16
srun --cpus-per-task=$SLURM_CPUS_PER_TASK singularity run \
	--rocm -B $SCRATCH:$SCRATCH \
	/scratch/project_id/docker/mujo_gfoot_env_v2.sif \
	/bin/sh -c \
	"
export PYTHONUSERBASE=/scratch/project_id/docker/venvs/spmarl4; \
python ./train/train_vmas_balance.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--scenario_name ${scenario_name} --n_rollout_threads 25 --n_eval_rollout_threads 24 --n_training_threads 1 \
--num_mini_batch 1 --episode_length 200 --use_ReLU --use_partial_obs \
--num_env_steps ${num_env_steps} --ppo_epoch 10 --use_eval --eval_episodes 32 --teacher ${teacher} --target_mean ${target_mean} --perf_lb ${perf_lb}
"

