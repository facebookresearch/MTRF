from copy import deepcopy

from ray import tune
import numpy as np
import os
from softlearning.misc.utils import get_git_rev, deep_update

try:
    import metaworld
except:
    print("Must install metaworld to access its envs")

DEFAULT_KEY = "__DEFAULT_KEY__"

# M = number of hidden units per layer
# N = number of hidden layers
M = 256
N = 2

REPARAMETERIZE = True
NUM_COUPLING_LAYERS = 2


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, ) * N,
        'squash': True,
        'observation_keys': None,
        'goal_keys': None,
        'observation_preprocessors_params': {}
    }
}


ALGORITHM_PARAMS_BASE = {
    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_n_episodes': 1,
        'eval_deterministic': False,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'save_training_video_frequency': 5,
        'eval_render_kwargs': {
            'width': 480,
            'height': 480,
            'mode': 'rgb_array',
        },
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'n_initial_exploration_steps': int(1e3),
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'verbose': True,

            'eval_n_episodes': 1,

            'ext_reward_coeff': 1,
            # 'rnd_int_rew_coeff': tune.grid_search([0, 1]),
            # 'normalize_ext_reward_gamma': tune.grid_search([0.99, 1]),
            'rnd_int_rew_coeff': tune.grid_search([0]),
        },
        'rnd_params': {
            'convnet_params': {
                'conv_filters': (16, 32, 64),
                'conv_kernel_sizes': (3, 3, 3),
                'conv_strides': (2, 2, 2),
                'normalization_type': None,
            },
            'fc_params': {
                'hidden_layer_sizes': (256, 256),
                'output_size': 512,
            },
            'kwargs': {
                # 'observation_keys': ("object_qpos", "object_qvel", "sawyer_pos"),
            }
        }
    },
    'MultiSAC': {
        'type': 'MultiSAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'n_initial_exploration_steps': int(1e3),
            'action_prior': 'uniform',
            'her_iters': tune.grid_search([0]),
            'eval_n_episodes': 3,

            # ======= R3L (Perturbation Controller + RND) =======
            # 'rnd_int_rew_coeffs': tune.grid_search([[0, 1], [1, 1]]),
            'rnd_int_rew_coeffs': [0, 1],
            'ext_reward_coeffs': [1, 0],

            # ======= Reset Controller + RND =======
            # 'rnd_int_rew_coeffs': [1, 1],
            # 'ext_reward_coeffs': [1, 1],

            # ======= Reset Controller =======
            # 'rnd_int_rew_coeffs': tune.grid_search([[0, 0], [1, 1]]),
            # 'rnd_int_rew_coeffs': [0, 0],
            # 'ext_reward_coeffs': [1, 1],

            # 'normalize_ext_reward_gamma': 0.99,
            'share_pool': False,
        },
        'rnd_params': {
            'convnet_params': {
                'conv_filters': (16, 32, 64),
                'conv_kernel_sizes': (3, 3, 3),
                'conv_strides': (2, 2, 2),
                'normalization_type': None,
            },
            'fc_params': {
                'hidden_layer_sizes': (256, 256),
                'output_size': 512,
            },
            'kwargs': {
                # RND reset controller should focus on maximizing count-based
                # bonus on object state/velocity
                'observation_keys': tune.grid_search([("object_xyz", ), None]),
            },
        },
    },
    'PhasedSAC': {
        'type': 'PhasedSAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'n_initial_exploration_steps': int(1e3),
            'action_prior': 'uniform',
            'eval_n_episodes': 1,

            # True at the phase index for perturbation
            # 'use_env_count_bonus_per_phase': [False, False],
            'use_env_count_bonus_per_phase': [False, False, False, False],
            # 'use_env_count_bonus_per_phase': [False, False, False, False, False],

            # 'use_env_count_bonus_per_phase': [False, False, True, False],
            # 'use_env_count_bonus_per_phase': [False, False, False, False, True],

            'phase_stop_training_iteration': [None, None, None, None],
            'rnd_int_rew_coeffs': [0] * 4,
            'ext_reward_coeffs': [1] * 4,

            # 'normalize_ext_reward_gamma': 0.99,
            'share_pool': False,
        },
        # 'rnd_params': {
        #     'convnet_params': {
        #         'conv_filters': (16, 32, 64),
        #         'conv_kernel_sizes': (3, 3, 3),
        #         'conv_strides': (2, 2, 2),
        #         'normalization_type': None,
        #     },
        #     'fc_params': {
        #         'hidden_layer_sizes': (256, 256),
        #         'output_size': 512,
        #     },
        #     'kwargs': {
        #         # RND reset controller should focus on maximizing count-based
        #         # bonus on object state/velocity
        #         'observation_keys': ("object_xyz", "object_euler", "object_qvel"),
        #     },
        # },
    },
}

MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 100,
    'gym': {
        DEFAULT_KEY: 100,
        'Point2D': {
            DEFAULT_KEY: 200,
        },
        'Pusher2D': {
            DEFAULT_KEY: 100,
            'Simple-v0': 150,
            'Test-v0': 150,
        },
        # Metaworld tasks
        # **{
        #     k[:k.rfind("-")]: { DEFAULT_KEY: 150 }
        #     for k in metaworld.MT1.ENV_NAMES
        # },
        'MiniGrid': {
            DEFAULT_KEY: 50,
        },
        'DClaw': {
            DEFAULT_KEY: 50,
            'TurnFixed-v0': 50,
            # 'TurnResetFree-v0': 100,
            'TurnResetFree-v0': 50,
            'TurnResetFreeSwapGoal-v0': tune.grid_search([100]),
            'TurnResetFreeRandomGoal-v0': 100,
            'TurnFreeValve3Fixed-v0': tune.grid_search([50]),
            # 'TurnFreeValve3RandomReset-v0': 50,
            'TurnFreeValve3ResetFree-v0': tune.grid_search([100]),
            'TurnFreeValve3ResetFreeSwapGoal-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeComposedGoals-v0': tune.grid_search([150]),

            # Translating Tasks
            'TranslatePuckFixed-v0': 50,
            'TranslateMultiPuckFixed-v0': 100,

            'TranslatePuckResetFree-v0': 50,

            # Lifting Tasks
            'LiftDDFixed-v0': tune.grid_search([50]),
            'LiftDDResetFree-v0': tune.grid_search([50]),

            # Flipping Tasks
            'FlipEraserFixed-v0': tune.grid_search([50]),
            'FlipEraserResetFree-v0': tune.grid_search([50]),
            'FlipEraserResetFreeSwapGoal-v0': tune.grid_search([50]),

            # Sliding Tasks
            'SlideBeadsFixed-v0': tune.grid_search([25]),
            'SlideBeadsResetFree-v0': tune.grid_search([25]),
            'SlideBeadsResetFreeEval-v0': tune.grid_search([25]),
        },
        'SawyerDhandInHandPickUp': {
            DEFAULT_KEY: 100,
        },
        'SawyerDhandInHandReach': {
            DEFAULT_KEY: 100,
        },
        'SawyerDhandInHandReposition': {
            DEFAULT_KEY: 100,
        },
        'SawyerDhandInHandDodecahedron': {
            DEFAULT_KEY: 100,
            'BasketFixed-v0': 100,
            'BasketResetFree-v0': 100,
            'BasketPhased-v0': 100,
            'BulbFixed-v0': 50,
            'BulbPhased-v0': 100,
            'FlipUpFixed-v0': 50,
        },
        'SawyerDhandInHandValve3': {
            DEFAULT_KEY: 100,
            'FlipUpResetFree-v0': 50,
            'FlipUpFixed-v0': 50,
            'FlipDownFixed-v0': 50,
        },
        'SawyerDhandInHandRod': {
            DEFAULT_KEY: 100,
        },
        'SawyerDhandInHandDumbbell': {
            DEFAULT_KEY: 100,
        },
        'SawyerDhandInHandMug': {
            DEFAULT_KEY: 100,
        },
        'SawyerDhandInHandPipe': {
            DEFAULT_KEY: 100,
        },
        'PincerManipulateResetFreePhasedSAC': {
            DEFAULT_KEY: 100,
            'v0': 100,
            'v1': 100
        },
        'PincerManipulateResetFreePhasedSACEval': {
            DEFAULT_KEY: 100,
            'v0': 100,
            'v1': 100
        },
        'PincerPickUpFixed': {
            DEFAULT_KEY: 100,
            'v0': 100,
        },
        'PincerFillYellow': {
            DEFAULT_KEY: 100,
            'v0': 100,
        },
        'PincerPullYellow': {
            DEFAULT_KEY: 100,
            'v0': 100,
        },
    },
}


NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 500,
    'gym': {
        DEFAULT_KEY: 200,
        'SawyerDhandInHandDodecahedron': {
            DEFAULT_KEY: 500,
            'PickupFixed-v0': 750,
            'AllPhasesResetFree-v1': 1000,

            'BasketFixed-v0': 500,
            'BasketResetController-v0': 1000,
            'BasketResetFree-v0': 1500,
            'BasketPhased-v0': 1000,

            'BulbPhased-v0': 1000,
            'BulbResetController-v0': 1000,
            'BulbFixed-v0': 1000,
            'BulbResetFree-v0': 1000,
        },
       'PincerManipulateResetFreePhasedSAC': {
            DEFAULT_KEY: 2000,
            'v0': 2000,
            'v1': 2000
        },
        'PincerManipulateResetFreePhasedSACEval': {
            DEFAULT_KEY: 2000,
            'v0': 2000,
            'v1': 2000
        },
        'PincerPickUpFixed': {
            DEFAULT_KEY: 2000,
            'v0': 2000,
        },
        'PincerFillYellow': {
            DEFAULT_KEY: 2000,
            'v0': 2000,
        },
        'PincerPullYellow': {
            DEFAULT_KEY: 2000,
            'v0': 2000,
        },
    },
}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE = {
    'gym': {
        "SawyerDhandInHandDodecahedron": {
            "RepositionRandomInit-v0": {
                'random_angle': tune.grid_search([True, False]),
                'readjust_to_object_in_reset': tune.grid_search([True, False]),
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
                'observation_keys': ("qp", "qv", "mocap_pos", "mocap_quat"),
                'reward_keys_and_weights': {
                    'object_to_target_xy_distance_reward': 1.0,
                    'object_to_hand_xyz_distance_reward': 1.0,
                    'span_dist': 1.0,
                },
            },
            "RepositionRandomInitEval-v0": {
                'random_angle': tune.sample_from(
                    lambda spec: spec.get('config')
                    ["environment_params"]
                    ["training"]
                    ["kwargs"]
                    ["random_angle"]
                ),
                'readjust_to_object_in_reset': tune.sample_from(
                    lambda spec: spec.get('config')
                    ["environment_params"]
                    ["training"]
                    ["kwargs"]
                    ["readjust_to_object_in_reset"]
                ),
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
                'observation_keys': ("qp", "qv", "mocap_pos", "mocap_quat"),
                'reward_keys_and_weights': {
                    'object_to_target_xy_distance_reward': 1.0,
                    'object_to_hand_xyz_distance_reward': 1.0,
                    'span_dist': 1.0,
                },
            },
            "PickupFixed-v0": {
                'reset_every_n_episodes': 1,
                'reset_offset': tune.grid_search([
                    np.array([-0.15, 0, 0.2]),
                    np.array([-0.15, 0, 0.175]),
                ]),
                'readjust_to_object_in_reset': True,
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
            },
            "PickupFixedEval-v0": {
                'reset_every_n_episodes': 1,
                'reset_offset': tune.sample_from(
                    lambda spec: spec.get('config')
                    ["environment_params"]
                    ["training"]
                    ["kwargs"]
                    ["reset_offset"]
                ),
                'readjust_to_object_in_reset': True,
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
            },

            # Testing basket task without the hoop to see if it's a dynamics issue
            "PalmDownRepositionMidairFixed-v0": {
                'reset_every_n_episodes': 1,
                'target_xyz_range_params': {
                    'type': 'DiscreteRange',
                    'values': [
                        np.array([0.92, 0.15, 1.2]),
                    ],
                },
            },
            "PalmDownRepositionMidairFixedEval-v0": {
                'reset_every_n_episodes': 1,
                'target_xyz_range_params': {
                    'type': 'DiscreteRange',
                    'values': [
                        np.array([0.92, 0.15, 1.2]),
                    ],
                },
            },

            # Basket task
            "BasketFixed-v0": {
                'reset_every_n_episodes': 1,
            },
            # "BasketFixedEval-v0": {
            #     'reset_every_n_episodes': 1,
            # },
            "BasketResetFree-v0": {
            },
            "BasketFixedEval-v0": {
                'reset_every_n_episodes': 1,
                'reset_offset': np.array([-0.15, 0, 0.125]),
            },
            "BasketPhased-v0": {
                'reset_every_n_episodes': np.inf,
                'perturb_off': tune.grid_search([False]),

                # Algorithm controls phase changes
                'commanded_phase_changes': True,
                # Disable these forced resets
                'max_episodes_in_phase': np.inf,
                'max_episodes_stuck': np.inf,
            },
            "BasketPhasedEval-v0": {
                'commanded_phase_changes': True,
                'reset_every_n_episodes': 1,
            },
            "BasketResetController-v0": {
                'reset_every_n_episodes': np.inf,
                'commanded_phase_changes': True,
            },
            "BasketResetControllerEval-v0": {
                'reset_every_n_episodes': 1,
                'commanded_phase_changes': True,
            },

            # Bulb task
            "BulbFixed-v0": {
                'reset_every_n_episodes': 1,
            },
            "BulbFixedEval-v0": {
                'reset_every_n_episodes': 1,
            },
            "BulbPhased-v0": {
                'reset_every_n_episodes': np.inf,

                # Algorithm controls phase changes
                'commanded_phase_changes': True,

                # Disable these forced resets
                'max_episodes_in_phase': np.inf,
                'max_episodes_stuck': np.inf,
            },
            "BulbPhasedEval-v0": {
                'reset_every_n_episodes': 1,
                'commanded_phase_changes': True,
            },

            "BulbResetFree-v0": {
                'reset_every_n_episodes': np.inf,
            },

            "BulbResetController-v0": {
                'reset_every_n_episodes': np.inf,
                'commanded_phase_changes': True,
            },
            "BulbResetControllerEval-v0": {
                'reset_every_n_episodes': 1,
                'commanded_phase_changes': True,
            },

            # Flip Up
            "FlipUpFixed-v0": {
                'reset_every_n_episodes': 1,
            },
            "FlipUpFixedEval-v0": {
                'reset_every_n_episodes': 1,
            },

            # Full phased
            "AllPhasesResetFree-v1": {
                'reset_every_n_episodes': np.inf,

                # Algorithm controls phase changes
                'commanded_phase_changes': True,
                # Disable these forced resets
                'max_episodes_in_phase': np.inf,
                'max_episodes_stuck': np.inf,

                'readjust_to_object_in_reset': True,
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
            },
            "AllPhasesResetFreeEval-v1": {
                'reset_every_n_episodes': 1,

                # Algorithm controls phase changes
                'commanded_phase_changes': True,
                # Disable these forced resets
                'max_episodes_in_phase': np.inf,
                'max_episodes_stuck': np.inf,

                'readjust_to_object_in_reset': True,
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
            },

            # Dodecahedron center --> random
            "RepositionCenterToRandom-v0": {
                'readjust_to_object_in_reset': tune.grid_search([True, False]),
                'n_bins': tune.grid_search([50, 100]),
                'commanded_phase_changes': True,
                'max_episodes_in_phase': np.inf,
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
            },
            "RepositionCenterToRandomEval-v0": {
                'readjust_to_object_in_reset': tune.sample_from(
                    lambda spec: spec.get('config')
                    ["environment_params"]
                    ["training"]
                    ["kwargs"]
                    ["readjust_to_object_in_reset"]
                ),
                'n_bins': tune.sample_from(
                    lambda spec: spec.get('config')
                    ["environment_params"]
                    ["training"]
                    ["kwargs"]
                    ["n_bins"]
                ),
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
                'commanded_phase_changes': True,
                'reset_every_n_episodes': 1,
                'max_episodes_in_phase': np.inf,
            },
            "RepositionCornerToCorner-v0": {
                'reset_every_n_episodes': 1,
                'readjust_to_object_in_reset': tune.grid_search([True, False]),
                'commanded_phase_changes': True,
                'max_episodes_in_phase': np.inf,
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
                'reward_keys_and_weights': {
                    'object_to_target_xy_distance_reward': 1.0,
                    'object_to_hand_xyz_distance_reward': 1.0,
                    'span_dist': 1.0,
                },
            },
            "RepositionCornerToCornerEval-v0": {
                'readjust_to_object_in_reset': tune.sample_from(
                    lambda spec: spec.get('config')
                    ["environment_params"]
                    ["training"]
                    ["kwargs"]
                    ["readjust_to_object_in_reset"]
                ),
                'readjust_hand_xyz': True,
                "readjust_hand_euler": False,
                'commanded_phase_changes': True,
                'reset_every_n_episodes': 1,
                'max_episodes_in_phase': np.inf,
                'reward_keys_and_weights': {
                    'object_to_target_xy_distance_reward': 1.0,
                    'object_to_hand_xyz_distance_reward': 1.0,
                    'span_dist': 1.0,
                },
            },
        },
    },
}


def get_num_epochs(universe, domain, task):
    level_result = NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_initial_exploration_steps(spec):
    config = spec.get('config', spec)
    initial_exploration_steps = 50 * (
        config
        ['sampler_params']
        ['kwargs']
        ['max_path_length']
    )

    return initial_exploration_steps


def get_checkpoint_frequency(spec):
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // NUM_CHECKPOINTS

    return checkpoint_frequency


def get_policy_params(universe, domain, task):
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


def get_algorithm_params(universe, domain, task):
    algorithm_params = {
        'kwargs': {
            'n_epochs': get_num_epochs(universe, domain, task),
            'n_initial_exploration_steps': tune.sample_from(
                get_initial_exploration_steps),
        }
    }

    return algorithm_params


def get_environment_params(universe, domain, task, from_vision):
    if from_vision:
        raise NotImplementedError
    else:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE

    environment_params = (
        params.get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params


NUM_CHECKPOINTS = 10
SAMPLER_PARAMS_PER_DOMAIN = {
    'DClaw': {
        'type': 'SimpleSampler',
    },
}


def get_variant_spec_base(universe, domain, task, task_eval, policy, algorithm, from_vision):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        get_algorithm_params(universe, domain, task),
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task, from_vision),
            },
            'evaluation': {
                'domain': domain,
                'task': task_eval,
                'universe': universe,
                'kwargs': (
                    tune.sample_from(lambda spec: (
                        spec.get('config', spec)
                        ['environment_params']
                        ['training']
                        .get('kwargs')
                    ))
                    if task == task_eval
                    else get_environment_params(universe, domain, task_eval, from_vision)),
            },
        },
        'policy_params': get_policy_params(universe, domain, task),
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, ) * N,
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
                'observation_preprocessors_params': {}
            },
            # 'discrete_actions': False,
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1e6),
            },
        },
        'sampler_params': deep_update({
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['sampler_params']['kwargs']['max_path_length']
                )),
                'batch_size': 256,
                'store_last_n_paths': 60,
        }
        }, SAMPLER_PARAMS_PER_DOMAIN.get(domain, {})),
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    env_kwargs = variant_spec['environment_params']['training']['kwargs']
    env_obs_keys = env_kwargs.get('observation_keys', tuple())
    env_goal_keys = env_kwargs.get('goal_keys', tuple())

    if not from_vision:
        non_pixel_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_pixel_obs_keys

    if env_goal_keys:
        variant_spec['policy_params']['kwargs']['goal_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['goal_keys'] = variant_spec[
                'Q_params']['kwargs']['goal_keys'] = env_goal_keys

    return variant_spec


IMAGE_ENVS = (
    ('robosuite', 'InvisibleArm', 'FreeFloatManipulation'),
)

def is_image_env(universe, domain, task, variant_spec):
    return ('image' in task.lower()
            or 'image' in domain.lower()
            or 'pixel_wrapper_kwargs' in (
                variant_spec['environment_params']['training']['kwargs'])
            or (universe, domain, task) in IMAGE_ENVS)


STATE_PREPROCESSOR_PARAMS = {
    'None': None,
}


PIXELS_PREPROCESSOR_PARAMS = {
    'ConvnetPreprocessor': tune.grid_search([
        {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (8, 16, 32),
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': tune.sample_from([None]),
                'downsampling_type': 'conv',
            },
        }
        for normalization_type in (None, )
    ]),
}


def get_variant_spec_image(universe,
                           domain,
                           task,
                           task_eval,
                           policy,
                           algorithm,
                           from_vision,
                           preprocessor_type,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe,
        domain,
        task,
        task_eval,
        policy,
        algorithm,
        from_vision,
        *args, **kwargs)

    if from_vision and is_image_env(universe, domain, task, variant_spec):
        assert preprocessor_type in PIXELS_PREPROCESSOR_PARAMS or preprocessor_type is None
        if preprocessor_type is None:
            preprocessor_type = "ConvnetPreprocessor"
        preprocessor_params = PIXELS_PREPROCESSOR_PARAMS[preprocessor_type]

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, ) * N
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )
    elif preprocessor_type:
        # Assign preprocessor to all parts of the state
        assert preprocessor_type in STATE_PREPROCESSOR_PARAMS
        preprocessor_params = STATE_PREPROCESSOR_PARAMS[preprocessor_type]
        obs_keys = variant_spec['environment_params']['training']['kwargs'].get('observation_keys', tuple())

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, ) * N
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                key: deepcopy(preprocessor_params)
                for key in obs_keys
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )

    return variant_spec


def get_variant_spec(args):
    universe, domain, task, task_eval = (
        args.universe,
        args.domain,
        args.task,
        args.task_evaluation)

    from_vision = args.vision
    preprocessor_type = args.preprocessor_type

    variant_spec = get_variant_spec_image(
        universe,
        domain,
        task,
        task_eval,
        args.policy,
        args.algorithm,
        from_vision,
        preprocessor_type)

    # if args.checkpoint_replay_pool is not None:
    variant_spec['run_params']['checkpoint_replay_pool'] = (
        args.checkpoint_replay_pool or False)

    return variant_spec
