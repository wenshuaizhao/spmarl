import sys
sys.path.append("../")
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import MatrixShareSelfPacedSubprocVecEnv, ShareDummyVecEnv
from deep_sprl.make_teacher import make_teacher



"""Train script for Matrix."""


def make_train_env(all_args, teacher=None):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "matrix":
                from envs.matrix.matrix_env import ClimbingEnv
                env = ClimbingEnv(all_args)
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
        return MatrixShareSelfPacedSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)], teacher=teacher, args=all_args)


def make_eval_env(all_args, teacher=None):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "matrix":
                from envs.matrix.matrix_env import ClimbingEnv
                env = ClimbingEnv(all_args)
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
        return MatrixShareSelfPacedSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)], teacher=teacher, args=all_args)


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='permutation',
                        help="Which task to run on")
    parser.add_argument("--max_num_agents", type=int, default=20)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--lower_context_bound", type=int, default=2)
    parser.add_argument("--upper_context_bound", type=int, default=20)
    parser.add_argument("--init_mean", type=float, default=6)
    parser.add_argument("--init_var", type=float, default=16)
    parser.add_argument("--target_mean", type=float, default=20)
    parser.add_argument("--target_var", type=float, default=4e-3)
    parser.add_argument("--std_lower_bound", type=float, default=0.2)
    parser.add_argument("--context_kl_threshold", type=float, default=8000)
    parser.add_argument("--max_kl", type=float, default=0.05)
    parser.add_argument("--perf_lb", type=float, default=0.5)
    parser.add_argument('--teacher', type=str,
                        default='no_teacher', choices=['sprl', 'random', 'linear', 'invlinear', 'no_teacher', 'spmarl', 'alpgmm', 'vacl'], help="observation range")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.seed = np.random.randint(1000, 10000)
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        # can or cannot use recurrent network?
        print("using", all_args.algorithm_name, 'without recurrent network')
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
        0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name+"_spmarl4",
                         entity='wszhao_aalto',
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.scenario_name,) +
                         "_seed" + str(all_args.seed),
                         #  group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         tags=["icml2025"],
                         reinit=True)
        all_args = wandb.config  # for wandb sweep
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.teacher is not None:
        teacher = make_teacher(teacher=all_args.teacher, args=all_args)
    else:
        teacher = None
        raise RuntimeError('Please specify the teacher')
    # env
    envs = make_train_env(all_args, teacher=teacher)
    eval_envs = make_eval_env(
        all_args, teacher=teacher) if all_args.use_eval else None
    max_num_agents = all_args.max_num_agents
    num_agents = all_args.max_num_agents
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "max_num_agents": max_num_agents,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.matrix_runner import MatrixRunner as Runner
    else:
        from runner.separated.matrix_runner import MatrixRunner as Runner

    if all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        from runner.separated.matrix_runner import MatrixRunner as Runner

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
