from collections import defaultdict

import numpy as np
from flatten_dict import flatten, unflatten
import imageio
from softlearning.models.utils import flatten_input_structure
from .base_sampler import BaseSampler
import os

class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._save_training_video_frequency = 0
        self._images = []
        self._algorithm = None

    @property
    def _policy_input(self):
        try:
            observation = flatten_input_structure({
                key: self._current_observation[key][None, ...]
            for key in self.policy.observation_keys
            })
        except Exception:
            from pprint import pprint; import ipdb; ipdb.set_trace(context=30)

        return observation

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        if (self._algorithm is not None
                and self._algorithm._save_training_video_frequency > 0
                and self._algorithm._epoch % self._algorithm._save_training_video_frequency == 0):
            if not hasattr(self, "_images"):
                self._images = []
            self._images.append(
                self.env.render(mode='rgb_array', width=256, height=256))

        action = self.policy.actions_np(self._policy_input)[0]
        next_observation, reward, terminal, info = self.env.step(action)

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in flatten(processed_sample).items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = unflatten({
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            })

            self.pool.add_path({
                key: value
                for key, value in last_path.items()
            })

            if (self._algorithm is not None
                    and self._algorithm._save_training_video_frequency > 0
                    and self._algorithm._epoch % self._algorithm._save_training_video_frequency == 0):
                self._last_n_paths.appendleft({
                    'images': self._images,
                    **last_path,
                })
            else:
                self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self.pool.terminate_episode()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)
            self._images = []

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size

        return self.pool.random_batch(batch_size, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics

    def set_save_training_video_frequency(self, flag):
        self._save_training_video_frequency = flag

    def __getstate__(self):
        state = super().__getstate__()
        state['_last_n_paths'] = type(state['_last_n_paths'])((
            type(path)((
                (key, value)
                for key, value in path.items()
                if key != 'images'
            ))
            for path in state['_last_n_paths']
        ))

        del state['_images']
        return state
