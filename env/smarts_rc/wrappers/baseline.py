from collections import deque
from typing import Sequence
import copy
import numpy as np
import gym
from scipy.spatial import distance
from smarts.core.controllers import ActionSpaceType

from tools.utils import batch_dicts
from env.smarts_rc.wrappers import common
from env.utils import compute_aid2uids


class FrameStack(gym.Wrapper):
    """By default, this wrapper will stack 3 consecutive frames as an agent observation"""

    def __init__(self, env, config):
        super(FrameStack, self).__init__(env)

        config.setdefault('frame_stack', 3)
        config.setdefault('goal_relative_pos', True)
        config.setdefault('distance_to_center', True)
        config.setdefault('speed', True)
        config.setdefault('steering', True)
        config.setdefault('heading_errors', [20, 'continuous'])
        config.setdefault('neighbor', 8)

        config.setdefault('action_type', 1)

        config.setdefault('collision_penalty', -50)
        config.setdefault('offroad_penalty', -50)
        config.setdefault('goal_reward', 20)

        self.config = config

        self.max_episode_steps = config.max_episode_steps
        self.num_stack = config.frame_stack

        self.frame_space = gym.spaces.Dict(common.subscribe_features(**config))
        self.observation_space = FrameStack.get_observation_space(
            self.frame_space, self.num_stack
        )
        self.observation_adapter = FrameStack.get_observation_adapter(
            self.observation_space, config
        )
        self.action_adapter = FrameStack.get_action_adapter(
            ActionSpaceType(config.action_type)
        )
        self.info_adapter = common.agent_info_adapter
        self.reward_adapter = FrameStack.get_reward_adapter(
            self.observation_adapter, config
        )

        self.agent_keys = list(self.env.agent_specs.keys())
        # self._last_observations = {k: None for k in self.agent_keys}

        self.frames = {
            agent_id: deque(maxlen=self.num_stack) for agent_id in self.agent_keys
        }

        self.n_agents = len(self.agent_keys)
        self.n_units = len(self.agent_keys)
        self.uid2aid = tuple(np.arange(self.n_agents))
        self.aid2uids = compute_aid2uids(self.uid2aid)

        if config.action_type == 1: 
            self.action_space = [gym.spaces.Discrete(4) 
                for _ in range(self.n_agents)]
            self.action_shape = [a.shape for a in self.action_space]
            self.action_dim = [a.n for a in self.action_space]
            self.action_dtype = [np.int32 for a in self.action_space]
            self.is_action_discrete = [True
                for _ in range(self.n_agents)]
        else:
            self.action_space = [gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32,
            ) for _ in range(self.n_agents)]
            self.action_shape = [a.shape for a in self.action_space]
            self.action_dim = [a.shape[0] for a in self.action_space]
            self.action_dtype = [np.float32 for a in self.action_space]
            self.is_action_discrete = [False
                for _ in range(self.n_agents)]

    def random_action(self):
        if self.config.action_type == 1:
            return {aid: np.random.randint(4) for aid in self.agent_keys}
        else:
            return {aid: np.random.uniform(-1, 1, (3,)) for aid in self.agent_keys}
        
    @staticmethod
    def get_observation_space(observation_space, frame_stack):
        if isinstance(observation_space, gym.spaces.Box):
            return gym.spaces.Tuple([observation_space] * frame_stack)
        elif isinstance(observation_space, gym.spaces.Dict):
            # inner_spaces = {}
            # for k, space in observation_space.spaces.items():
            #     inner_spaces[k] = FrameStack.get_observation_space(space, wrapper_config)
            # dict_space = gym.spaces.Dict(spaces)
            return gym.spaces.Tuple([observation_space] * frame_stack)
        else:
            raise TypeError(
                f"Unexpected observation space type: {type(observation_space)}"
            )

    @staticmethod
    def get_action_space(action_space, wrapper_config=None):
        return action_space

    @staticmethod
    def get_observation_adapter(
        observation_space, feature_configs, wrapper_config=None
    ):
        def func(env_obs_seq):
            assert isinstance(env_obs_seq, Sequence)
            observation = common.cal_obs(env_obs_seq, observation_space, feature_configs)
            return observation

        return func

    @staticmethod
    def get_action_adapter(action_type):
        return common.ActionAdapter.from_type(action_type)

    @staticmethod
    def stack_frames(frames):
        proto = frames[0]

        if isinstance(proto, dict):
            res = dict()
            for key in proto.keys():
                res[key] = np.stack([frame[key] for frame in frames], axis=0)
        elif isinstance(proto, np.ndarray):
            res = np.stack(frames, axis=0)
        else:
            raise NotImplementedError

        return res

    def _get_observations(self, raw_frames):
        """Update frame stack with given single frames,
        then return nested array with given agent ids
        """

        for k, frame in raw_frames.items():
            self.frames[k].append(frame)

        agent_ids = list(raw_frames.keys())
        observations = dict.fromkeys(agent_ids)

        for k in agent_ids:
            observation = list(self.frames[k])
            observation = self.observation_adapter(observation)
            observations[k] = observation

        return observations

    def _get_rewards(self, env_observations, env_rewards):
        agent_ids = list(env_rewards.keys())
        rewards = dict.fromkeys(agent_ids, None)

        for k in agent_ids:
            rewards[k] = self.reward_adapter(list(self.frames[k]), env_rewards[k])
        return rewards

    def _get_infos(self, env_obs, rewards, infos):
        if self.info_adapter is None:
            return infos

        res = {}
        agent_ids = list(env_obs.keys())
        for k in agent_ids:
            res[k] = self.info_adapter(env_obs[k], rewards[k], infos[k])
        return res

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self.action_adapter(action)
            for agent_id, action in agent_actions.items()
        }
        env_observations, env_rewards, dones, infos = super().step(
            agent_actions
        )
        # assert len(agent_actions) == len(env_observations), (len(agent_actions), len(env_observations), dones, len(infos))

        observations = self._get_observations(env_observations)
        rewards = self._get_rewards(env_observations, env_rewards)
        infos = self._get_infos(env_observations, env_rewards, infos)
        # self._update_last_observation(self.frames)

        return observations, rewards, dones, infos

    def reset(self):
        observations = super(FrameStack, self).reset()
        for k, observation in observations.items():
            _ = [self.frames[k].append(observation) for _ in range(self.num_stack)]
        # self._update_last_observation(self.frames)
        return self._get_observations(observations)

    @staticmethod
    def get_reward_adapter(observation_adapter, config):
        def func(env_obs_seq, env_reward):
            penalty, bonus = 0.0, 0.0
            obs_seq = observation_adapter(env_obs_seq)

            # ======== Penalty: too close to neighbor vehicles
            # if the mean ttc or mean speed or mean dist is higher than before, get penalty
            # otherwise, get bonus
            last_env_obs = env_obs_seq[-1]
            neighbor_features_np = np.asarray([e.get("neighbor") for e in obs_seq])
            if neighbor_features_np is not None:
                new_neighbor_feature_np = neighbor_features_np[-1].reshape((-1, 5))
                mean_dist = np.mean(new_neighbor_feature_np[:, 0])
                mean_ttc = np.mean(new_neighbor_feature_np[:, 2])

                last_neighbor_feature_np = neighbor_features_np[-2].reshape((-1, 5))
                mean_dist2 = np.mean(last_neighbor_feature_np[:, 0])
                # mean_speed2 = np.mean(last_neighbor_feature[:, 1])
                mean_ttc2 = np.mean(last_neighbor_feature_np[:, 2])
                penalty += (
                    0.03 * (mean_dist - mean_dist2)
                    # - 0.01 * (mean_speed - mean_speed2)
                    + 0.01 * (mean_ttc - mean_ttc2)
                )

            # ======== Penalty: distance to goal =========
            goal = last_env_obs.ego_vehicle_state.mission.goal
            ego_2d_position = last_env_obs.ego_vehicle_state.position[:2]
            goal_position = getattr(goal, "position", ego_2d_position)[:2]
            goal_dist = distance.euclidean(ego_2d_position, goal_position)
            penalty += -0.01 * goal_dist

            old_obs = env_obs_seq[-2]
            old_goal = old_obs.ego_vehicle_state.mission.goal
            old_ego_2d_position = old_obs.ego_vehicle_state.position[:2]
            old_goal_position = getattr(old_goal, "position", old_ego_2d_position)[:2]
            old_goal_dist = distance.euclidean(old_ego_2d_position, old_goal_position)
            penalty += 0.1 * (old_goal_dist - goal_dist)  # 0.05

            # ======== Penalty: distance to the center
            distance_to_center_np = np.asarray(
                [e["distance_to_center"] for e in obs_seq]
            )
            diff_dist_to_center_penalty = np.abs(distance_to_center_np[-2]) - np.abs(
                distance_to_center_np[-1]
            )
            penalty += 0.01 * diff_dist_to_center_penalty[0]

            # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
            ego_events = last_env_obs.events
            # ::collision
            penalty += config.collision_penalty if len(ego_events.collisions) > 0 else 0.0
            # ::off road
            penalty += config.offroad_penalty if ego_events.off_road else 0.0
            # ::reach goal
            if ego_events.reached_goal:
                bonus += config.goal_reward

            # ::reached max_episode_step
            if ego_events.reached_max_episode_steps:
                penalty += -0.5
            else:
                bonus += 0.5

            # ======== Penalty: heading error penalty
            # if obs.get("heading_errors", None):
            #     heading_errors = obs["heading_errors"][-1]
            #     penalty_heading_errors = -0.03 * heading_errors[:2]
            #
            #     heading_errors2 = obs["heading_errors"][-2]
            #     penalty_heading_errors += -0.01 * (heading_errors[:2] - heading_errors2[:2])
            #     penalty += np.mean(penalty_heading_errors)

            # ======== Penalty: penalise sharp turns done at high speeds =======
            if last_env_obs.ego_vehicle_state.speed > 60:
                steering_penalty = -pow(
                    (last_env_obs.ego_vehicle_state.speed - 60)
                    / 20
                    * last_env_obs.ego_vehicle_state.steering
                    / 4,
                    2,
                )
            else:
                steering_penalty = 0
            penalty += 0.1 * steering_penalty

            # ========= Bonus: environment reward (distance travelled) ==========
            bonus += 0.05 * env_reward
            return bonus + penalty

        return func

    # def _update_last_observation(self, observations):
    #     for agent_id, obs in observations.items():
    #         self._last_observations[agent_id] = copy.copy(obs)


class NormalizeAndFlatten(gym.Wrapper):
    def __init__(self, env, to_single_agent=True):
        super().__init__(env)

        self.agent_keys = sorted(self.env.agent_specs.keys())
        self.left_agents = set(self.agent_keys)
        self._info = {}

        self.to_single_agent = to_single_agent
        if to_single_agent:
            self.n_agents = 1
            self.n_units = len(self.env.agent_specs)
            self.uid2aid = tuple(np.zeros(self.n_units, np.int32))
            self.aid2uids = compute_aid2uids(self.uid2aid)
        else:
            self.n_agents = self.env.n_agents
            self.n_units = len(self.env.agent_specs)
            self.uid2aid = tuple(np.arange(self.n_units))
            self.aid2uids = compute_aid2uids(self.uid2aid)
        self.use_sample_mask = True
        obs_spaces = self.observation_space
        obs_dim = len(obs_spaces) * sum([o.shape[0] for o in obs_spaces[0].spaces.values()])
        self.obs_dim = obs_dim
        self.obs_shape = [dict(
            obs=(obs_dim,), 
            global_state=(obs_dim,), 
            sample_mask=()
        ) for _ in range(self.n_agents)]
        self.obs_dtype = [dict(
            obs=np.float32, 
            global_state=np.float32, 
            sample_mask=np.float32, 
        ) for _ in range(self.n_agents)]

        self._dense_score = np.zeros(self.n_agents, np.float32)
        self._score = np.zeros(self.n_agents, np.float32)
        self._epslen = 0

    def random_action(self):
        action = self.env.random_action()
        action = np.stack([action[aid] for aid in self.agent_keys])
        return action

    def reset(self):
        self._dense_score = np.zeros(self.n_units, np.float32)
        self._score = np.zeros(self.n_units, np.float32)
        self._epslen = 0

        obs = self.env.reset()
        self.left_agents = set(obs)
        return self.observation(obs)

    def step(self, action):
        action = self.action(action)
        obs, reward, done, info = self.env.step(action)
        self.left_agents = set([aid for aid, d in done.items() if not d])
        obs = self.observation(obs)
        self.update_stats(reward, info)
        reward = self.reward(reward)
        done = self.done(done)
        info = self.info(info)
        info['game_over'] = np.all(done)
        return obs, reward, done, info

    def observation(self, observations):
        agent_obs = []
        for aid in self.agent_keys:
            if aid in observations:
                # normalize obs
                obs = batch_dicts(observations[aid], np.concatenate) 
                obs['distance_to_center'] /= 1.5
                obs['goal_relative_pos'] /= 50
                # obs['heading_errors'] /= 1 # already normalizeed by sin
                obs['neighbor'] = (obs['neighbor'] - 30) / 20
                obs['speed'] /= 50
                # obs['steering'] /= 1       # already normalized

                # concatenate obs and add the unit dimension
                obs = np.concatenate(list(obs.values())).astype(np.float32)
                obs = np.expand_dims(obs, 0)

                agent_obs.append({
                    'obs': obs, 
                    'global_state': obs, 
                    'sample_mask': np.ones(1, np.float32)
                })
            else:
                obs = np.zeros((1, self.obs_dim), np.float32)
                agent_obs.append({
                    'obs': obs, 
                    'global_state': obs, 
                    'sample_mask': np.zeros(1, np.float32)
                })
        if self.to_single_agent:
            agent_obs = [batch_dicts(agent_obs, np.concatenate)]

        return agent_obs

    def reward(self, reward):
        new_reward = []
        for aid in self.agent_keys:
            new_reward.append(np.array(
                [reward[aid] if aid in reward else 0], np.float32))
        if self.to_single_agent:
            new_reward = [np.concatenate(new_reward)]
        return new_reward

    def done(self, done):
        new_done = []
        for aid in self.agent_keys:
            new_done.append(np.array(
                [done[aid] if aid in done else True], np.float32))
        if self.to_single_agent:
            new_done = [np.concatenate(new_done)]
        return new_done
    
    def info(self, info):
        new_info = []
        for aid in self.agent_keys:
            if aid in info:
                self._info[aid] = info[aid]
            new_info.append(self._info[aid])
        return batch_dicts(new_info)

    def action(self, action):
        new_action = {}
        for aid, a in zip(self.agent_keys, action):
            if aid in self.left_agents:
                new_action[aid] = a
        return new_action

    def update_stats(self, reward, info):
        self._epslen += 1
        for i, aid in enumerate(self.agent_keys):
            if aid in reward:
                self._dense_score[i] += reward[aid]
            if aid in info:
                self._score[i] += info[aid]['goal_score']
                info[aid]['score'] = self._score[i]
                info[aid]['dense_score'] = self._dense_score[i]
                info[aid]['epslen'] = self._epslen
