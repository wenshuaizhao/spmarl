#!/usr/bin/env python
import sys
sys.path.append("../")
from deep_sprl.make_teacher import make_teacher
from envs.env_wrappers import CMASShareSelfPacedSubprocVecEnv, ShareDummyVecEnv
from envs.vmas.vmas_env import ContextualVMASEnv
from config import get_config
import torch
from pathlib import Path
import numpy as np
import setproctitle
import socket
import wandb
import os

# wandb.login(key='42e7d207d3284a031abccd90fec5d448784c4bad')

"""Train script for MPEs."""


def make_train_env(all_args, teacher=None):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "VMAS":
                env = ContextualVMASEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return CMASShareSelfPacedSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)], teacher=teacher, args=all_args, eval=False)


def make_eval_env(all_args, teacher=None):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "VMAS":
                env = ContextualVMASEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return CMASShareSelfPacedSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)], teacher=teacher, args= all_args, eval=True)


def parse_args(args, parser):
    parser.add_argument('--lower_context_bound', type=float,
                        default=2, help="lower context bound")
    parser.add_argument('--upper_context_bound', type=float,
                        default=10, help="lower context bound")
    parser.add_argument('--init_mean', type=float,
                        default=5, help="lower context bound")
    parser.add_argument('--init_var', type=float,
                        default=25, help="lower context bound")
    parser.add_argument('--target_mean', type=float,
                        default=6, help="lower context bound")
    parser.add_argument('--target_var', type=float,
                        default=4e-3, help="lower context bound")
    parser.add_argument('--std_lower_bound', type=float,
                        default=0.2, help="lower context bound")
    parser.add_argument('--context_kl_threshold', type=float,
                        default=8000, help="lower context bound")
    parser.add_argument('--max_kl', type=float,
                        default=0.1, help="lower context bound")
    parser.add_argument('--perf_lb', type=float,
                        default=20, help="lower context bound")

    parser.add_argument('--scenario_name', type=str,
                        default='balance', help="Which scenario to run on")
    parser.add_argument('--continuous_actions', action='store_true',
                        default=False, help="set continuous actions")
    parser.add_argument('--max_num_agents', type=int,
                        default=10, help="number of players")
    parser.add_argument('--num_agents', type=int,
                        default=6, help="number of players") 
    parser.add_argument('--max_steps', type=int,
                        default=200, help="the meximum steps of the game")
    parser.add_argument('--teacher', type=str,
                        default=None, choices=['sprl', 'random', 'linear', 'no_teacher', 'spmarl', 'vacl', 'alpgmm', 'invlinear'], help="observation range")
    parser.add_argument('--sparse_reward', action='store_true',
                        default=False, help="observation range")
    parser.add_argument('--distance_threshold', type=float,
                        default=10, help="distance threshold for sparse reward")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    if all_args.seed_specify:
        all_args.seed = all_args.seed
    else:
        all_args.seed = np.random.randint(1000, 10000)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    cur_dir = Path(os.path.join(os.path.dirname(__file__)))
    result_dir = os.path.join(
        cur_dir.parent.absolute().parent.absolute(), "results")

    run_dir = Path(result_dir) / all_args.env_name / \
        all_args.scenario_name / all_args.algorithm_name
    w_run_dir = run_dir/'local_logs'
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name+"_spmarl4",
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         tags=[all_args.experiment_name],
                         reinit=True)
    else:
        if not w_run_dir.exists():
            os.makedirs(str(w_run_dir))
        from datetime import datetime
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        curr_run = f'run_{date_time}'
        run_dir = w_run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" +
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    if all_args.teacher is not None:
        teacher = make_teacher(teacher=all_args.teacher, args=all_args)
    else:
        teacher = None
        raise RuntimeError('Please specify the teacher')
    envs = make_train_env(all_args, teacher=teacher)
    eval_envs = make_eval_env(
        all_args, teacher=teacher) if all_args.use_eval else None
    num_agents = all_args.num_agents
    max_num_agents = all_args.max_num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "max_num_agents": max_num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.vmas_runner import VMASRunner as Runner
    else:
        raise NotImplementedError("It is not implemented yet.")

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
