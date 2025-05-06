
from .teachers.spl import SelfPacedTeacherV2 as SPRL
from .teachers.spl.spmarl_teacher import SPMARLTeacher as SPMARL
from .teachers.spl.vacl_teacher import VACLTeacher
from .teachers.spl.alpgmm_teacher import ALPGMMTeacher
import numpy as np

class LinearTeacher():
    def __init__(self, total_timestep, initial_context, target_context):
        self.total_timestep=total_timestep
        self.initial_context=initial_context
        self.target=target_context
        self.threshold=round(total_timestep*0.8)
        self.cur_context=self.initial_context
        self.teacher_name='linear'

    def update_distribution(self, cur_timestep, *args, **kwargs):
        if cur_timestep<= self.threshold:
            self.cur_context = (self.target - self.initial_context) * cur_timestep / self.threshold + \
                        self.initial_context
        else:
            self.cur_context= self.target
    def sample(self, size=1):
        return np.ones([size])*self.cur_context
    
class InvLinearTeacher():
    def __init__(self, total_timestep, initial_context, target_context):
        self.total_timestep=total_timestep
        self.initial_context=initial_context
        self.target=target_context
        self.threshold=round(total_timestep*0.8)
        self.cur_context=self.initial_context
        self.teacher_name='invlinear'

    def update_distribution(self, cur_timestep, *args, **kwargs):
        if cur_timestep<= self.threshold:
            self.cur_context = self.initial_context - (self.initial_context-self.target) * cur_timestep / self.threshold
        else:
            self.cur_context= self.target
    def sample(self, size=1):
        return np.ones([size])*self.cur_context
    
        
class FixedTeacher():
    def __init__(self, target_context=8) -> None:
        self.target=target_context
        self.teacher_name='no_teacher'
    def sample(self, size=1):
        return np.ones([size])*self.target
    def update_distribution(self, *args, **kwargs):
        pass
    
class RandomTeacher():
    def __init__(self, lower=6, upper=20, target_context=8) -> None:
        self.lower=lower
        self.upper=upper
        self.target=target_context
        self.teacher_name='random'
        self.std = np.max([target_context-lower, upper-target_context])
    def sample(self, size=1):
        # return np.random.uniform(low=self.lower, high=self.upper, size=size)
        return np.clip(np.random.normal(loc=self.target, scale=self.std, size=size), self.lower, self.upper)
    def update_distribution(self, *args, **kwargs):
        pass
    

        

def make_teacher(teacher=None, args=None):
    ### Macros for the curriculum learning
    LOWER_CONTEXT_BOUNDS = np.array([args.lower_context_bound])
    UPPER_CONTEXT_BOUNDS = np.array([args.upper_context_bound])

    INITIAL_MEAN = np.array([args.init_mean])
    INITIAL_VARIANCE = np.diag([args.init_var])

    TARGET_MEAN = np.array([args.target_mean])
    TARGET_VARIANCE = np.diag([args.target_var])

    STD_LOWER_BOUND = np.array([args.std_lower_bound])
    KL_THRESHOLD = args.context_kl_threshold
    MAX_KL = args.max_kl
    PERF_LB = args.perf_lb
    NUM_PARTICLEs=args.n_rollout_threads

    bounds = (LOWER_CONTEXT_BOUNDS.copy(), UPPER_CONTEXT_BOUNDS.copy())
    if teacher=='sprl':

        teacher = SPRL(TARGET_MEAN.copy(), TARGET_VARIANCE.copy(), INITIAL_MEAN.copy(),
                                    INITIAL_VARIANCE.copy(), bounds, PERF_LB,
                                    max_kl=MAX_KL, std_lower_bound=STD_LOWER_BOUND.copy(),
                                    kl_threshold=KL_THRESHOLD, use_avg_performance=True)
    elif teacher == 'linear':
        teacher=LinearTeacher(total_timestep=args.num_env_steps, initial_context=LOWER_CONTEXT_BOUNDS, target_context=TARGET_MEAN)
    elif teacher == 'invlinear':    
        teacher=InvLinearTeacher(total_timestep=args.num_env_steps, initial_context=UPPER_CONTEXT_BOUNDS, target_context=TARGET_MEAN)
    elif teacher == 'no_teacher':
        teacher=FixedTeacher(target_context=TARGET_MEAN)
    elif teacher == 'random':
        teacher= RandomTeacher(lower=LOWER_CONTEXT_BOUNDS, upper=UPPER_CONTEXT_BOUNDS, target_context=TARGET_MEAN)
    elif teacher == 'spmarl':
        teacher= SPMARL(TARGET_MEAN.copy(), TARGET_VARIANCE.copy(), INITIAL_MEAN.copy(),
                                    INITIAL_VARIANCE.copy(), bounds, PERF_LB,
                                    max_kl=MAX_KL, std_lower_bound=STD_LOWER_BOUND.copy(),
                                    kl_threshold=KL_THRESHOLD, use_avg_performance=True)
    elif teacher == 'vacl':
        teacher = VACLTeacher(TARGET_MEAN.copy(), TARGET_VARIANCE.copy(), INITIAL_MEAN.copy(),
                                    INITIAL_VARIANCE.copy(), bounds, NUM_PARTICLEs)
    elif teacher == 'alpgmm':
        teacher = ALPGMMTeacher(mins=LOWER_CONTEXT_BOUNDS, maxs=UPPER_CONTEXT_BOUNDS, target_context=TARGET_MEAN, fit_rate=args.n_rollout_threads)
        
    return teacher