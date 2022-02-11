import itertools
import numpy as np
import cv2
import gym
from env.typing import EnvOutput

from utility.utils import batch_dicts, dict2AttrDict, convert_batch_with_func
from env import make_env
from env.utils import batch_env_output


class Env(gym.Wrapper):
    def __init__(self, config, env_fn=make_env, agents={}):
        self.env = env_fn(config, eid=None, agents=agents)
        if 'seed' in config and hasattr(self.env, 'seed'):
            self.env.seed(config['seed'])
        self.name = config['env_name']
        self.max_episode_steps = self.env.max_episode_steps
        self.n_envs = getattr(self.env, 'n_envs', 1)
        self.env_type = 'Env'
        self._stats = self.env.stats()
        self._stats['n_workers'] = 1
        self._stats['n_envs'] = self.n_envs
        super().__init__(self.env)

    def reset(self, idxes=None):
        return self.env.reset()

    def random_action(self, *args, **kwargs):
        action = self.env.random_action() if hasattr(self.env, 'random_action') \
            else self.env.action_space.sample()
        return action
        
    def step(self, action, **kwargs):
        output = self.env.step(action, **kwargs)
        return output

    def stats(self):
        return dict2AttrDict(self._stats)

    def manual_reset(self):
        self.env.manual_reset()

    """ the following code is needed for ray """
    def score(self, *args):
        return self.env.score()

    def epslen(self, *args):
        return self.env.epslen()

    def mask(self, *args):
        return self.env.mask()

    def prev_obs(self):
        return self.env.prev_obs()

    def info(self, *args):
        return self.env.info(*args)

    def output(self):
        return self.env.output()

    def game_over(self):
        return self.env.game_over()

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            img = self.env.get_screen()
        else:
            img = self.env.render(mode='rgb_array')

        if size is not None and size != img.shape[:2]:
            # cv2 receive size of form (width, height)
            img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
            
        return img

    def record_default_state(self, aid, state):
        self.env.record_default_state(aid, state)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


class VecEnvBase(gym.Wrapper):
    def __init__(self):
        self.env_type = 'VecEnv'
        super().__init__(self.env)

    def stats(self):
        return dict2AttrDict(self._stats)

    def _get_idxes(self, idxes):
        if idxes is None:
            idxes = list(range(self.n_envs))
        elif isinstance(idxes, int):
            idxes = [idxes]
        return idxes
    
    def process_output(self, out, convert_batch=True):
        if convert_batch:
            return batch_env_output(out)
        else:
            return out


class VecEnv(VecEnvBase):
    def __init__(self, config, env_fn=make_env, agents={}):
        self.n_envs = n_envs = config.pop('n_envs', 1)
        self.name = config['env_name']
        self.envs = [env_fn(
            config, config['eid'] + eid if 'eid' in config else None, agents)
            for eid in range(n_envs)]
        self.env = self.envs[0]
        if 'seed' in config:
            [env.seed(config['seed'] + i) 
                for i, env in enumerate(self.envs)
                if hasattr(env, 'seed')]
        self.max_episode_steps = self.env.max_episode_steps
        super().__init__()
        self._stats = self.env.stats()
        self._stats['n_workers'] = 1
        self._stats['n_envs'] = self.n_envs
        self._stats['is_multi_agent'] = False

    def random_action(self, *args, **kwargs):
        return np.stack([env.random_action() if hasattr(env, 'random_action') \
            else env.action_space.sample() for env in self.envs])

    def reset(self, idxes=None, convert_batch=True, **kwargs):
        idxes = self._get_idxes(idxes)
        out = [self.envs[i].reset() for i in idxes]

        return self.process_output(out, convert_batch=convert_batch)

    def step(self, actions, convert_batch=True, **kwargs):
        if isinstance(actions, (tuple, list)):
            actions = zip(*actions)
        outs = [e.step(a) for e, a in zip(self.envs, actions)]

        return self.process_output(outs, convert_batch=convert_batch)

    def manual_reset(self):
        [e.manual_reset() for e in self.envs]

    def score(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].score() for i in idxes]

    def epslen(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].epslen() for i in idxes]

    def mask(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return np.stack([self.envs[i].mask() for i in idxes])

    def game_over(self):
        return np.stack([env.game_over() for env in self.envs])

    def prev_obs(self, idxes=None):
        idxes = self._get_idxes(idxes)
        obs = [self.envs[i].prev_obs() for i in idxes]
        if isinstance(obs[0], dict):
            obs = batch_dicts(obs)
        return obs

    def info(self, idxes=None, convert_batch=False):
        idxes = self._get_idxes(idxes)
        info = [self.envs[i].info() for i in idxes]
        if convert_batch:
            info = batch_dicts(info)
        return info

    def output(self, idxes=None, convert_batch=True):
        idxes = self._get_idxes(idxes)
        out = [self.envs[i].output() for i in idxes]

        return self.process_output(out, convert_batch=convert_batch)

    def get_screen(self, size=None, convert_batch=True):
        if hasattr(self.env, 'get_screen'):
            imgs = [env.get_screen() for env in self.envs]
        else:
            imgs = [env.render(mode='rgb_array') for env in self.envs]

        if size is not None:
            # cv2 receive size of form (width, height)
            imgs = [cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
                            for i in imgs]
        if convert_batch:
            imgs = np.stack(imgs)

        return imgs

    def record_default_state(self, aids, states):
        state_type = type(states)
        states = [state_type(*s) for s in zip(*states)]
        for e, a, s in zip(self.envs, aids, states):
            e.record_default_state(a, s)

    def _envvec_op(self, name, convert_batch=True, **kwargs):
        method = lambda e: getattr(e, name)
        if kwargs:
            kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_envs)]
                if isinstance(v, np.ndarray) else list(zip(*v)) 
                for k, v in kwargs.items()}
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
                for k, v in kwargs.items()])]
            out = [method(env)(**kw) for env, kw in zip(self.envs, kwargs)]
        else:
            out = [method(env)() for env in self.envs]

        return self.process_output(out, convert_batch=convert_batch)

    def close(self):
        if hasattr(self.env, 'close'):
            [env.close() for env in self.envs]


class MAVecEnv(VecEnvBase):
    def __init__(self, config, env_fn=make_env, agents={}):
        self.n_envs = n_envs = config.pop('n_envs', 1)
        self.name = config['env_name']
        self.envs = [env_fn(
            config, config['eid'] + eid if 'eid' in config else None, agents)
            for eid in range(n_envs)]
        self.env = self.envs[0]
        if 'seed' in config:
            [env.seed(config['seed'] + i) 
                for i, env in enumerate(self.envs)
                if hasattr(env, 'seed')]
        self.max_episode_steps = self.env.max_episode_steps
        super().__init__()
        self._stats = self.env.stats()
        self._stats['n_workers'] = 1
        self._stats['n_envs'] = self.n_envs
        self._stats['is_multi_agent'] = True

    def random_action(self, *args, **kwargs):
        actions = list(zip(*[env.random_action() for env in self.envs]))
        return actions

    def reset(self, idxes=None, convert_batch=True, **kwargs):
        idxes = self._get_idxes(idxes)
        out = [self.envs[i].reset() for i in idxes]

        return self.process_output(out, convert_batch=convert_batch)

    def step(self, actions, convert_batch=True, **kwargs):
        if isinstance(actions, (tuple, list)):
            actions = zip(*actions)
        outs = [e.step(a) for e, a in zip(self.envs, actions)]

        return self.process_output(outs, convert_batch=convert_batch)

    def manual_reset(self):
        [e.manual_reset() for e in self.envs]

    def score(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].score() for i in idxes]

    def epslen(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].epslen() for i in idxes]

    def mask(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return np.stack([self.envs[i].mask() for i in idxes])

    def game_over(self):
        return np.stack([env.game_over() for env in self.envs])

    def prev_obs(self, idxes=None):
        idxes = self._get_idxes(idxes)
        obs = [self.envs[i].prev_obs() for i in idxes]
        if isinstance(obs[0], dict):
            obs = batch_dicts(obs)
        return obs

    def info(self, idxes=None, convert_batch=False):
        idxes = self._get_idxes(idxes)
        info = [self.envs[i].info() for i in idxes]
        if convert_batch:
            info = batch_dicts(info)
        return info

    def output(self, idxes=None, convert_batch=True):
        idxes = self._get_idxes(idxes)
        out = [self.envs[i].output() for i in idxes]

        return self.process_output(out, convert_batch=convert_batch)

    def get_screen(self, size=None, convert_batch=True):
        if hasattr(self.env, 'get_screen'):
            imgs = [env.get_screen() for env in self.envs]
        else:
            imgs = [env.render(mode='rgb_array') for env in self.envs]

        if size is not None:
            # cv2 receive size of form (width, height)
            imgs = [cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
                            for i in imgs]
        if convert_batch:
            imgs = np.stack(imgs)

        return imgs

    def record_default_state(self, aids, states):
        state_type = type(states)
        states = [state_type(*s) for s in zip(*states)]
        for e, a, s in zip(self.envs, aids, states):
            e.record_default_state(a, s)

    def _envvec_op(self, name, convert_batch=True, **kwargs):
        method = lambda e: getattr(e, name)
        if kwargs:
            kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_envs)]
                if isinstance(v, np.ndarray) else list(zip(*v)) 
                for k, v in kwargs.items()}
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
                for k, v in kwargs.items()])]
            out = [method(env)(**kw) for env, kw in zip(self.envs, kwargs)]
        else:
            out = [method(env)() for env in self.envs]

        return self.process_output(out, convert_batch=convert_batch)

    def close(self):
        if hasattr(self.env, 'close'):
            [env.close() for env in self.envs]

    def process_output(self, out, convert_batch=True):
        obs = []
        reward = []
        discount = []
        reset = []
        for o in out:
            obs.append(o.obs)
            reward.append(o.reward)
            discount.append(o.discount)
            reset.append(o.reset)
        assert len(obs) == self.n_envs, len(obs)
        assert len(reward) == self.n_envs, len(reward)
        assert len(discount) == self.n_envs, len(discount)
        assert len(reset) == self.n_envs, len(reset)
        if convert_batch:
            obs = [convert_batch_with_func(o) for o in zip(*obs)]
            reward = [convert_batch_with_func(r) for r in zip(*reward)]
            discount = [convert_batch_with_func(d) for d in zip(*discount)]
            reset = [convert_batch_with_func(r) for r in zip(*reset)]
            out = EnvOutput(obs, reward, discount, reset)

        else:
            obs = list(zip(*obs))
            reward = list(zip(*reward))
            discount = list(zip(*discount))
            reset = list(zip(*reset))
            out = EnvOutput(obs, reward, discount, reset)
        assert len(out.obs) == self.env.n_agents, len(out.obs)
        assert len(out.reward) == self.env.n_agents, len(out.reward)
        assert len(out.discount) == self.env.n_agents, len(out.discount)
        assert len(out.reset) == self.env.n_agents, len(out.reset)
        return out


if __name__ == '__main__':
    config = dict(
        env_name='smac_6h_vs_8z',
        n_workers=8,
        n_envs=1,
        use_state_agent=True,
        use_mustalive=True,
        add_center_xy=True,
        timeout_done=True,
        add_agent_id=False,
        obs_agent_id=False,
    )
    env = Env(config)
    for k in range(100):
        o, r, d, re = env.step(env.random_action())
        print(k, d, re, o['episodic_mask'])
        print(r, env.score(), env.epslen())
