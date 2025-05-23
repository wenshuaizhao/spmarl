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
from envs.env_wrappers import ShareSelfPacedSubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv
from deep_sprl.make_teacher import make_teacher
"""Train script for SMAC."""

def parse_smacv2_distribution(args):
    units = args.units.split('v')
    distribution_config = {
        "n_units": int(units[0]),
        "n_enemies": int(units[1]),
        "start_positions": {
            "dist_type": "surrounded",
            "map_x": 32,
            "map_y": 32,
        }
    }
    if 'protoss' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "self_paced_weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        }
    elif 'zerg' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "self_paced_weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        } 
    elif 'terran' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "self_paced_weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        } 
    return distribution_config

def make_train_env(all_args, teacher=None):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2v2":
                from envs.starcraft2.SMACv2_modified import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            elif all_args.env_name == "SMAC":
                from envs.starcraft2.SMAC import SMAC
                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSelfPacedSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)], teacher=teacher, args=all_args)


def make_eval_env(all_args, teacher=None):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2v2":
                from envs.starcraft2.SMACv2_modified import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            elif all_args.env_name == "SMAC":
                from envs.starcraft2.SMAC import SMAC
                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSelfPacedSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)], teacher=teacher, args=all_args)


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='10gen_protoss',
                        help="Which smac map to run on")
    parser.add_argument('--units', type=str, default='15v5') # for smac v2
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--max_num_agents", type=int, default=15)
    parser.add_argument("--lower_context_bound", type=int, default=5)
    parser.add_argument("--upper_context_bound", type=int, default=15)
    parser.add_argument("--init_mean", type=float, default=10)
    parser.add_argument("--init_var", type=float, default=25)
    parser.add_argument("--target_mean", type=float, default=5)
    parser.add_argument("--target_var", type=float, default=4e-3)
    parser.add_argument("--std_lower_bound", type=float, default=0.2)
    parser.add_argument("--context_kl_threshold", type=float, default=8000)
    parser.add_argument("--max_kl", type=float, default=0.1)
    parser.add_argument("--perf_lb", type=float, default=0.6)
    parser.add_argument('--teacher', type=str,
                        default='no_teacher', choices=['sprl', 'random', 'linear', 'invlinear', 'no_teacher', 'spmarl', 'alpgmm', 'vacl'], help="observation range")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    if all_args.seed_specify:
        all_args.seed=all_args.seed
    else:
        all_args.seed=np.random.randint(1000,10000)
    print("seed is :",all_args.seed)

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
    elif all_args.algorithm_name == "happo"  or all_args.algorithm_name == "hatrpo":
        # can or cannot use recurrent network?
        print("using", all_args.algorithm_name, 'without recurrent network')
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True
    
    if all_args.teacher == "linear":
        all_args.lower_context_bound = all_args.upper_context_bound

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
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name+"_spmarl4",
                         entity='wszhao_aalto',
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.units) +
                              "_seed" + str(all_args.seed),
                        #  group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         tags=[all_args.experiment_name],
                         reinit=True)
        all_args = wandb.config # for wandb sweep
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

    # env init
    if all_args.teacher is not None:
        teacher=make_teacher(teacher=all_args.teacher, args=all_args)
    else:
        teacher=None
        raise RuntimeError('Please specify the teacher') 
    # env
    envs = make_train_env(all_args, teacher=teacher)
    eval_envs = make_eval_env(all_args, teacher=teacher) if all_args.use_eval else None

    if all_args.env_name == "SMAC":
        from smac.env.starcraft2.maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == 'StarCraft2':
        from envs.starcraft2.smac_maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == "SMACv2" or all_args.env_name == 'StarCraft2v2':
        from smacv2.env.starcraft2.maps import get_map_params
        num_agents = parse_smacv2_distribution(all_args)['n_units']
    max_num_agents=all_args.max_num_agents
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
        from runner.shared.smac_runner import SMACRunner as Runner
    else:
        from runner.separated.smac_runner import SMACRunner as Runner

    if all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        from runner.separated.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
