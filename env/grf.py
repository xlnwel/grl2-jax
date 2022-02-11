import numpy as np
import gym
import gfootball.env as football_env

from env.utils import compute_aid2uids


class GRF:
    def __init__(
        self,
        # built-in configs for grf
        env_name,
        representation='simple115',
        rewards='scoring,checkpoints',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=1000,
        logdir='data/grf',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        # required configs for grl
        max_episode_steps=3000,
        use_action_mask=True,
        uid2aid=None,
        **kwargs,
    ):
        self.name = env_name

        self.env = football_env.create_environment(
            self.name, 
            representation=representation,
            rewards=rewards,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir=logdir,
            extra_players=extra_players,
            number_of_left_players_agent_controls=number_of_left_players_agent_controls,
            number_of_right_players_agent_controls=number_of_right_players_agent_controls,
        )

        self.max_episode_steps = max_episode_steps

        self.use_action_mask = use_action_mask  # if action mask is used
        self.use_life_mask = False              # if life mask is used

        if uid2aid is None:
            uid2aid = tuple(np.zeros(number_of_left_players_agent_controls, dtype=np.int32)) \
                + tuple(np.ones(number_of_right_players_agent_controls, dtype=np.int32))
        self.uid2aid = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

        assert number_of_left_players_agent_controls + number_of_right_players_agent_controls \
            == self.n_units, \
                (self.uid2aid, number_of_left_players_agent_controls, number_of_right_players_agent_controls)

        self.action_space = [
            self.env.action_space[0] 
            if isinstance(self.env.action_space, gym.spaces.MultiDiscrete) 
            else self.env.action_space 
            for _ in range(self.n_agents)]
        self.action_shape = [a.shape for a in self.action_space]
        self.action_dim = [a.n for a in self.action_space]
        self.action_dtype = [np.int32 for _ in self.action_space]
        self.is_action_discrete = [isinstance(a, gym.spaces.Discrete) for a in self.action_space]

        self.obs_shape = self._get_observation_shape()
        self.obs_dtype = self._get_observation_dtype()

        # The following stats should be updated in self.step and be reset in self.reset
        # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
        self._score = np.zeros(self.n_units, dtype=np.float32)
        # The accumulated episodic rewards we give to the agent. It includes shaped rewards
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        # The length of the episode
        self._epslen = 0

    def _get_observation_shape(self):
        obs_shape = self.env.observation_space.shape \
            if self.n_units == 1 else self.env.observation_space.shape[1:]
        shape = [dict(
            obs=obs_shape,
            global_state=obs_shape,
        ) for _ in range(self.n_agents)]

        if self.use_action_mask:
            for aid in range(self.n_agents):
                shape[aid]['action_mask'] = (self.action_space[aid].n,)

        return shape

    def _get_observation_dtype(self):
        obs_dtype = self.env.observation_space.dtype
        dtype = [dict(
            obs=obs_dtype,
            global_state=obs_dtype,
        ) for _ in range(self.n_agents)]

        if self.use_action_mask:
            for aid in range(self.n_agents):
                dtype[aid]['action_mask'] = bool

        return dtype

    def random_action(self):
        action = [[] for _ in self.aid2uids]
        for aid in self.uid2aid:
            action[aid].append(self.action_space[0].sample())
        action = [np.array(a, dtype=np.int32) for a in action]
        return action

    def reset(self):
        obs = self.env.reset()

        self._score = np.zeros(self.n_units, dtype=np.float32)
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        self._epslen = 0

        return self._get_obs(obs)

    def step(self, action):
        action = np.concatenate(action)
        obs, reward, done, info = self.env.step(action)

        rewards = np.reshape(reward, -1)
        dones = np.tile(done, self.n_units)

        self._epslen += 1
        self._dense_score += rewards
        self._score += rewards > 0 if done else 0

        info = dict(
            score=self._score,
            dense_score=self._dense_score,
            epslen=self._epslen,
            game_over=done
        )

        agent_obs = self._get_obs(obs)
        agent_rewards = [np.reshape(rewards[uids], -1) for uids in self.aid2uids]
        agent_dones = [np.reshape(dones[uids], -1) for uids in self.aid2uids]

        return agent_obs, agent_rewards, agent_dones, info

    def close(self):
        return self.env.close()

    def _get_obs(self, obs):
        if self.n_units == 1:
            obs = np.expand_dims(obs, 0)
            agent_obs = [dict(
                obs=obs,
                global_state=obs,
                action_mask=np.ones((1, self.action_dim[0]), bool)
            )]
        else:
            agent_obs = [dict(
                obs=obs[uids],
                global_state=obs[uids],
                action_mask=np.ones((len(uids), self.action_dim[aid]), bool)
            ) for aid, uids in enumerate(self.aid2uids)]

        return agent_obs


if __name__ == '__main__':
    config = {
        'env_name': 'academy_counterattack_hard',
        'number_of_left_players_agent_controls': 1,
        'number_of_right_players_agent_controls': 0,
        'uid2aid': None
    }

    from utility.display import print_dict_tensors
    env = GRF(**config)
    env.reset()
    o, r, d, i = env.step(env.random_action())
    for oo in o:
        print_dict_tensors(oo)
    for rr in r:
        print(rr)
    for dd in d:
        print(dd)