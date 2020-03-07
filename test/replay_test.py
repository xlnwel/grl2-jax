import numpy as np
import gym

from replay.func import create_replay
from replay.ds.sum_tree import SumTree
from utility.utils import set_global_seed


n_steps = 1
bs = 100
gamma = .99
capacity = 3000
config = dict(
    type='uniform',
    n_steps=n_steps,
    gamma=gamma,
    batch_size=bs,
    min_size=7,
    capacity=capacity,
    has_next_state=True,
)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, config):
        self.min_size = max(10000, config['batch_size']*10)
        self.batch_size = config['batch_size']
        self.ptr, self.size, self.max_size = 0, 0, int(float(config['capacity']))
        self._type = 'uniform'
        print('spinup')

    def buffer_type(self):
        return self._type

    def good_to_learn(self):
        return self.ptr >= self.min_size

    def add(self, state, action, reward, next_state, done, **kwargs):
        if not hasattr(self, 'state'):
            obs_dim = state.shape[0]
            act_dim = action.shape[0]
            self.state = np.zeros([self.max_size, obs_dim], dtype=np.float32)
            self.next_state = np.zeros([self.max_size, obs_dim], dtype=np.float32)
            self.action = np.zeros([self.max_size, act_dim], dtype=np.float32)
            self.reward = np.zeros(self.max_size, dtype=np.float32)
            self.done = np.zeros(self.max_size, dtype=np.bool)
            self.n_steps = np.ones(self.max_size, dtype=np.uint8)
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return dict(state=self.state[idxs],
                    next_state=self.next_state[idxs],
                    action=self.action[idxs],
                    reward=self.reward[idxs],
                    done=self.done[idxs],
                    steps=self.n_steps[idxs])


class TestClass:
    def test_buffer_op(self):
        replay = create_replay(config)
        simp_replay =  ReplayBuffer(config)

        env = gym.make('BipedalWalkerHardcore-v2')

        s = env.reset()
        for i in range(10000):
            a = env.action_space.sample()
            ns, r, d, _ = env.step(a)
            if d:
                ns = env.reset()
            replay.add(state=s.astype(np.float32), action=a.astype(np.float32), reward=np.float32(r), next_state=ns.astype(np.float32), done=d)
            simp_replay.add(state=s, action=a, reward=r, next_state=ns, done=d)
            s = ns

            if i > 1000:
                set_global_seed(i)
                sample1 = replay.sample()
                set_global_seed(i)
                sample2 = simp_replay.sample()

                for k in sample1.keys():
                    np.testing.assert_allclose(sample1[k], sample2[k], err_msg=f'{k}')

    def test_sum_tree(self):
        for _ in range(10):
            cap = np.random.randint(10, 20)
            st1 = SumTree(cap)
            st2 = SumTree(cap)

            # test update
            bs = np.random.randint(5, cap+1)
            idxes = np.unique(np.random.randint(cap, size=bs))
            priorities = np.random.randint(10, size=len(idxes))
            for idx, p in zip(idxes, priorities):
                st1.update(idx, p)

            st2.batch_update(idxes, priorities)
            np.testing.assert_equal(st1.container, st2.container)
        
            # test find
            bs = np.random.randint(2, bs)
            intervals = np.linspace(0, st1.total_priorities, bs+1)
            values = np.random.uniform(intervals[:-1], intervals[1:])

            priorities1, indexes1 = list(zip(*[st1.find(v) for v in values]))
            priorities2, indexes2 = st2.batch_find(values)

            np.testing.assert_equal(priorities1, priorities2)
            np.testing.assert_equal(indexes1, indexes2)

            # validate sum tree's internal structure
            nodes = np.arange(st1.tree_size)
            left, right = 2 * nodes + 1, 2 * nodes + 2
            np.testing.assert_equal(st1.container[nodes], st1.container[left] + st1.container[right])
            np.testing.assert_equal(st2.container[nodes], st2.container[left] + st2.container[right])
            