import argparse
from distutils.util import strtobool
import json
import os
import pickle
import glob
import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.misc.utils import save_video
import numpy as np


DEFAULT_RENDER_KWARGS = {
    # 'mode': 'human',
    'mode': 'rgb_array',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=100)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-kwargs', '-r',
                        type=json.loads,
                        default='{}',
                        help="Kwargs for rollouts renderer.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")
    parser.add_argument('--use-state-estimator',
                        type=lambda x: bool(strtobool(x)),
                        default=False)


    args = parser.parse_args()

    return args


def simulate_policy(args):
    session = tf.keras.backend.get_session()
    checkpoint_path = args.checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    checkpoint_paths = [
        checkpoint_dir
        for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_path, 'checkpoint_*')),
                                     key=lambda d: float(d.split("checkpoint_")[1]))
    ]

    dump_dir = os.path.join(experiment_path, 'evaluations/')
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    all_paths = []
    for checkpoint_dir in checkpoint_paths[::2]:

        with session.as_default():
            pickle_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        environment_params = (
            variant['environment_params']['evaluation']
            if 'evaluation' in variant['environment_params']
            else variant['environment_params']['training'])

        environment_params['kwargs']['device_path'] = '/dev/ttyUSB0'
        environment_params['kwargs']['camera_config'] = {'topic': '/kinect2_001144463747/qhd/image_color', 'image_shape': (256, 256, 3)}
        environment_params['kwargs']['init_pos_range'] = list(
            (np.array([0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi, np.pi/4, np.pi/2, np.pi*3/4]) + (-75 * np.pi/180)) % (2*np.pi) - np.pi
        )
        environment_params['kwargs']['target_pos_range'] = [-75*np.pi/180]
        environment_params['kwargs']['cycle_inits'] = True

        evaluation_environment = get_environment_from_params(environment_params)

        policy = (
            get_policy_from_variant(variant, evaluation_environment))

        policy_weights = picklable['policy_weights']
        if variant['algorithm_params']['type'] in ['MultiSAC', 'MultiVICEGAN']:
            policy_weights = policy_weights[0]
        policy.set_weights(policy_weights)
        # dump_path = os.path.join(checkpoint_path, 'policy_params.pkl')
        # with open(dump_path, 'wb') as f:
        #     pickle.dump(picklable['policy_weights'], f)

        render_kwargs = {**DEFAULT_RENDER_KWARGS, **args.render_kwargs}

        with policy.set_deterministic(args.deterministic):
            paths = rollouts(args.num_rollouts,
                             evaluation_environment,
                             policy,
                             path_length=args.max_path_length,
                             render_kwargs=render_kwargs)

        if render_kwargs.get('mode') == 'rgb_array':
            fps = 2 // getattr(evaluation_environment, 'dt', 1/30)
            for i, path in enumerate(paths):
                # video_save_dir = os.path.expanduser('/tmp/simulate_policy/')
                video_save_path = os.path.join(checkpoint_dir, f'episode_{i}.mp4')

                save_video(path['images'], video_save_path, fps=fps)
        all_paths.append(paths)

    with open(os.path.join(dump_dir, 'evaluation_paths.pkl'), 'wb') as f:
        pickle.dump(all_paths, f)
    return paths


if __name__ == '__main__':
    args = parse_args()
    paths = simulate_policy(args)
