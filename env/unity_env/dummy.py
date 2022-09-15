from typing import List
import numpy as np
import gym

# from env.utils import compute_aid2uids

"""  Naming Conventions:
An Agent is the one that makes decisions, it's typically a RL agent
A Unit is a controllable unit in the environment.
Each agent may control a number of units.
All ids start from 0 and increases sequentially
"""

def compute_aid2uids(uid2aid):
    """ Compute aid2uids from uid2aid """
    aid2uids = []
    for uid, aid in enumerate(uid2aid):
        if aid > len(aid2uids):
            raise ValueError(f'uid2aid({uid2aid}) is not sorted in order')
        if aid == len(aid2uids):
            aid2uids.append((uid, ))
        else:
            aid2uids[aid] += (uid,)
    aid2uids = [np.array(uids, np.int32) for uids in aid2uids]

    return aid2uids


# MAGIC NUMBERS
# TODO: overwrite them using stats from the environment and delete these global variables
ACTION_DIM = 5

# NOTE: I comment out Unity-related code for the purpose of simple tests.

# NOTE: Keep the class name fixed; do not invent a new one!
# We do not rely on this for distinction!
class UnityEnv:
    def __init__(
        self, 
        uid2aid,
        n_envs,
        unity_config,
        # expand kwargs for your environment 
        **kwargs
    ):
        # uid2aid is a list whose indices are the unit ids and values are agent ids. 
        # It specifies which agent controlls the unit. 
        # We expect it to be sorted in the consecutive ascending order
        # That is, [0, 1, 1] is valid. [0, 1, 0] and [0, 0, 2] are invalid
        self.uid2aid: list = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_agents = len(self.aid2uids)	# the number of agents
        self.n_units = len(self.uid2aid)

        unity_config['n_envs'] = n_envs
        # from env.unity_env.interface import UnityInterface
        # self.env = UnityInterface(**unity_config)

        # the number of envs running in parallel, which must be the same as the number 
        # of environment copies specified when compiing the Unity environment 
        self.n_envs = n_envs
        # The same story goes with n_units
        self.n_units = len(uid2aid)  # the number of units

        # The maximum number of steps per episode; 
        # the length of an episode should never exceed this value
        self.max_episode_steps = kwargs['max_episode_steps']

        self.use_action_mask = True # if action mask is used
        self.use_life_mask = True   # if life mask is used

        self.action_space = [gym.spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents)]
        self.action_dim = [a.n for a in self.action_space]
        self.action_dtype = [a.dtype for a in self.action_space]

        # We expect <obs> in self.reset and self.step to return a list of dicts,
        # each associated to an agent. self.obs_shape and self.obs_dtype specify 
        # the corresponding shape and dtype.
        # We do not consider the environment and player dimensions here!
        self.obs_shape = [dict(
            obs=self._get_obs_shape(aid),
            global_state=self._get_global_state_shape(aid),
        ) for aid in range(self.n_agents)]
        self.obs_dtype = [dict(
            obs=np.float32,
            global_state=np.float32,
        ) for _ in range(self.n_agents)]
        if self.use_life_mask:
            for aid in range(self.n_agents):
                # 1 if the unit is alive, otherwise 0
                self.obs_shape[aid]['life_mask'] = ()
                self.obs_dtype[aid]['life_mask'] = np.float32
        if self.use_action_mask:
            for aid in range(self.n_agents):
                # 1 if the action is valid, otherwise 0
                self.obs_shape[aid]['action_mask'] = (self.action_space[aid].n,)
                self.obs_dtype[aid]['action_mask'] = bool

        # The following stats should be updated in self.step and be reset in self.reset
        # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
        self._score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        # The accumulated episodic rewards we give to the agent. It includes shaped rewards
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        # The length of the episode
        self._epslen = np.zeros(self.n_envs, np.int32)

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.int32)
            actions.append(a)
        return actions

    def reset(self):
        self._score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._epslen = np.zeros(self.n_envs, np.int32)

        # decision_steps, terminal_steps = self.env.reset()
        decision_steps, terminal_steps = {}, {}

        return self._get_obs(decision_steps, terminal_steps)
    
    def step(self, actions):
        # TODO: auto-reset when done is True or when max_episode_steps meets
        # NOTE: this should be done environment-wise. 
        # It will not be an easy task; please take extra care to make it right!
        # action = self._set_action(behavior_names, action)
        # decision_steps, terminal_steps = self.env.step()
        decision_steps, terminal_steps = {}, {}

        agent_obs = self._get_obs(decision_steps, terminal_steps)
        reward = np.random.normal(size=self.n_envs)
        done = np.random.randint(0, 2, self.n_envs, dtype=bool)
        discount = 1 - done # we return discount instead of done

        assert reward.shape == (self.n_envs,), reward.shape
        assert discount.shape == (self.n_envs,), discount.shape
        # obtain ndarrays of shape (n_envs, n_units)
        rewards = np.tile(reward.reshape(-1, 1), (1, self.n_units))
        discounts = np.tile(discount.reshape(-1, 1), (1, self.n_units))
        
        # TODO: these stats should be updated accordingly when auto resetting
        self._epslen += 1
        self._dense_score += rewards
        self._score += np.where(discounts, 0, rewards>0)  # an example for competitive games
        reset = done
        assert reset.shape == (self.n_envs,), reset.shape
        resets = np.tile(reset.reshape(-1, 1), (1, self.n_units))
        
        self._info = [dict(
            score=self._score[i].copy(),
            dense_score=self._dense_score[i].copy(),
            epslen=self._epslen[i],
            game_over=discount[i] == 0
        ) for i in range(self.n_envs)]

        # group stats of units controlled by the same agent
        agent_reward = [rewards[:, uids] for uids in self.aid2uids]
        agent_discount = [discounts[:, uids] for uids in self.aid2uids]
        agent_reset = [resets[:, uids] for uids in self.aid2uids]

        # we return agent-wise data
        return agent_obs, agent_reward, agent_discount, agent_reset
    
    def info(self):
        return self._info

    def close(self):
        # close the environment
        pass

    def _get_obs_shape(self, aid):
        return ((aid+1) * 2,)

    def _get_global_state_shape(self, aid):
        return ((aid+1) * 1,)

    def _get_obs(self, decision_step, terminal_step):
        obs = [{} for _ in range(self.n_agents)]
        for aid in range(self.n_agents):
            for k in self.obs_shape[aid].keys():
                obs[aid][k] = np.zeros(
                    (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]), 
                    dtype=self.obs_dtype[aid][k]
                )
                assert obs[aid][k].shape == (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]), \
                    (obs[aid][k].shape, (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]))

        return obs

    def _set_action(self, names: List[str], actions: np.ndarray):
        raise NotImplementedError
        # pass


""" Test Code """
if __name__ == '__main__':
    config = dict(
        env_name='dummy',
        uid2aid=[0, 1, 1, 1, 1],
        max_episode_steps=100,
        n_envs=4,
        unity_config={},
    )
    # from utility.display import print_dict, print_dict_tensors
    def print_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict(v, prefix+'\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict(v._asdict(), prefix+'\t')
            else:
                print(f'{prefix} {k}: {v}')

    def print_dict_tensors(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict_tensors(v, prefix+'\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict_tensors(v._asdict(), prefix+'\t')
            else:
                print(f'{prefix} {k}: {v.shape} {v.dtype}')

    env = UnityEnv(**config)
    observations = env.reset()
    print('reset observations')
    for i, o in enumerate(observations):
        print_dict_tensors(o, f'\tagent{i}')
    for k in range(1, 3):
        actions = env.random_action()
        print(f'Step {k}, random actions', actions)
        observations, rewards, dones, reset = env.step(actions)
        print(f'Step {k}, observations')
        for i, o in enumerate(observations):
            print_dict_tensors(o, f'\tagent{i}')
        print(f'Step {k}, rewards', rewards)
        print(f'Step {k}, dones', dones)
        print(f'Step {k}, reset', reset)
        info = env.info()
        print(f'Step {k}, info')
        for aid, i in enumerate(info):
            print_dict(i, f'\tenv{aid}')
