import itertools
import numpy as np
import cv2
import gym

from utility.utils import batch_dicts, dict2AttrDict
from env import make_env
from env.utils import batch_env_output


class Env(gym.Wrapper):
    def __init__(self, config, env_fn=make_env, agents={}):
        self.env = env_fn(config, eid=None, agents=agents)
        if 'seed' in config and hasattr(self.env, 'seed'):
            self.env.seed(config['seed'])
        self.name = config['name']
        self.max_episode_steps = self.env.max_episode_steps
        self.n_envs = 1
        self.env_type = 'Env'
        self._stats = self.env.stats()
        self._stats['n_workers'] = 1
        self._stats['n_envs'] = 1
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

    """ the following code is needed for ray """
    def score(self, *args):
        return self.env.score()

    def epslen(self, *args):
        return self.env.epslen()

    def mask(self, *args):
        return self.env.mask()

    def prev_obs(self):
        return self.env.prev_obs()

    def info(self):
        return self.env.info()

    def output(self):
        return self.env.output()

    def game_over(self):
        return self.env.game_over()
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            img = self.env.get_screen()
        else:
            img = self.env.render(mode='rgb_array')

        if size is not None and size != img.shape[:2]:
            # cv2 receive size of form (width, height)
            img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
            
        return img


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
        self.name = config['name']
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

    def random_action(self, *args, **kwargs):
        return np.stack([env.random_action() if hasattr(env, 'random_action') \
            else env.action_space.sample() for env in self.envs])

    def reset(self, idxes=None, convert_batch=True, **kwargs):
        idxes = self._get_idxes(idxes)
        out = [self.envs[i].reset() for i in idxes]

        return self.process_output(out, convert_batch=convert_batch)

    def step(self, actions, convert_batch=True, **kwargs):
        return self._envvec_op('step', action=actions, 
            convert_batch=convert_batch, **kwargs)

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

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            imgs = np.stack([env.get_screen() for env in self.envs])
        else:
            imgs = np.stack([env.render(mode='rgb_array') for env in self.envs])

        if size is not None:
            # cv2 receive size of form (width, height)
            imgs = np.stack([cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
                            for i in imgs])
        
        return imgs

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


class TwoPlayerSequentialVecEnv(VecEnvBase):
    def __init__(self, config, other_ids, other_agent, env_fn=make_env):
        self.n_envs = n_envs = config.pop('n_envs', 1)
        self.name = config['name']
        self.envs = [env_fn(
            config, config['eid'] + eid if 'eid' in config else None)
            for eid in range(n_envs)]
        self.env = self.envs[0]
        if 'seed' in config:
            [env.seed(config['seed'] + i) 
                for i, env in enumerate(self.envs)
                if hasattr(env, 'seed')]
        self.other_ids = other_ids
        self.other_agent = other_agent

        self.max_episode_steps = self.env.max_episode_steps
        super().__init__()
        self._stats = self.env.stats()
        self._stats['n_workers'] = 1
        self._stats['n_envs'] = self.n_envs

    def set_agent(self, agent):
        """ link to agent for bookkeeping """
        self.agent = agent

    def set_other_agent(self, other_agent):
        self.other_agent = other_agent

    def random_action(self, *args, **kwargs):
        return np.stack([env.random_action() if hasattr(env, 'random_action') \
            else env.action_space.sample() for env in self.envs])

    def reset(self, idxes=None, convert_batch=True, **kwargs):
        idxes = self._get_idxes(idxes)
        outs = [self.envs[i].reset() for i in idxes]
        outs = self.step_other_players(outs)

        return self.process_output(outs, convert_batch=convert_batch)

    def step(self, actions, convert_batch=True):
        if isinstance(actions, (tuple, list)):
            actions = zip(*actions)
        outs = [e.step(a) for e, a in zip(self.envs, actions)]
        outs = self.step_other_players(outs)

        assert np.all([o.obs['pid'] not in self.other_ids for o in outs]), [o.obs['pid'] not in self.other_ids for o in outs]
        return self.process_output(outs, convert_batch=convert_batch)

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

    def step_other_players(self, outs):
        def extract_other_player_eids_outs(eids, outs):
            assert len(eids) == len(outs), (len(eids), len(outs))
            other_eids = []
            other_outs = []
            for i, o in zip(eids, outs):
                if o.obs['pid'] in self.other_ids:
                    other_eids.append(i)
                    other_outs.append(o)
            return other_eids, other_outs

        other_eids, other_outs = extract_other_player_eids_outs(list(range(self.n_envs)), outs)
        while other_eids:
            assert self.other_agent is not None, self.other_agent
            assert len(other_eids) == len(other_outs), (len(other_eids), len(other_outs))
            # assert False, f'Oops. Should not be there for now {other_eids}'
            batch_other_outs = batch_env_output(other_outs)
            for i, o in zip(other_eids, other_outs):
                if o.discount == 0:
                    info = self.envs[i].info()
                    self.agent.store(
                        score=info['score'], 
                        epslen=info['epslen'], 
                        win_rate=info['won'])
            actions, terms = self.other_agent(batch_other_outs)
            if isinstance(actions, (tuple, list)):
                actions = list(zip(*actions))
            assert len(other_eids) == len(actions), (other_eids, actions)
            next_outs = [
                self.envs[i].step(a) for i, a in zip(other_eids, actions)]
            assert len(other_eids) == len(next_outs), (len(other_eids), len(next_outs))
            for i, o in zip(other_eids, next_outs):
                if o.obs['pid'] not in self.other_ids:
                    outs[i] = o
            other_eids, other_outs = extract_other_player_eids_outs(other_eids, next_outs)
        for i, o in enumerate(outs):
            assert o.obs['pid'] not in self.other_ids, (i, o.obs['pid'])
        return outs

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            imgs = np.stack([env.get_screen() for env in self.envs])
        else:
            imgs = np.stack([env.render(mode='rgb_array') for env in self.envs])

        if size is not None:
            # cv2 receive size of form (width, height)
            imgs = np.stack([cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
                            for i in imgs])
        
        return imgs

    def close(self):
        if hasattr(self.env, 'close'):
            [env.close() for env in self.envs]


if __name__ == '__main__':
    config = dict(
        name='smac_6h_vs_8z',
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
