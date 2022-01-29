from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import gym
import numpy as np
from env.overcooked_env2.mdp.actions import Action
from env.overcooked_env2.mdp.overcooked_mdp import OvercookedGridworld
from env.overcooked_env2.mdp.overcooked_env import OvercookedEnv
from env.overcooked_env2.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS


class PlayerException(Exception):
    """ Raise when players in the environment are incorrectly set """


class MultiAgentEnv(gym.Env, ABC):
    """
    Base class for all Multi-agent environments.

    :param ego_ind: The player that the ego represents
    :param n_units: The number of players in the game
    :param partners: Lists of agents to choose from for the partner players
    """

    def __init__(self,
                 ego_ind: int = 0,
                 n_units: int = 2,
                 partners: Optional[List[List]] = None):
        self.ego_ind = ego_ind
        self.n_units = n_units
        if partners is not None:
            if len(partners) != n_units - 1:
                raise PlayerException(
                    "The number of partners needs to equal the number \
                    of non-ego players")

            for plist in partners:
                if not isinstance(plist, list) or not plist:
                    raise PlayerException(
                        "Sublist for each partner must be nonempty list")

        self.partners = partners or [[]] * (n_units - 1)
        self.partnerids = [0] * (n_units - 1)

        self._players: Tuple[int, ...] = tuple()
        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None

        self.should_update = [False] * (self.n_units - 1)
        self.total_rews = [0] * (self.n_units)
        self.ego_moved = False

    def getDummyEnv(self, player_num: int):
        """
        Returns a dummy environment with just an observation and action
        space that a partner agent can use to construct their policy network.

        :param player_num: the partner number to query
        """
        return self

    def _get_partner_num(self, player_num: int) -> int:
        if player_num == self.ego_ind:
            raise PlayerException(
                "Ego agent is not set by the environment")
        elif player_num > self.ego_ind:
            return player_num - 1
        return player_num

    def add_partner_agent(self, agent, player_num: int = 1) -> None:
        """
        Add agent to the list of potential partner agents. If there are
        multiple agents that can be a specific player number, the environment
        randomly samples from them at the start of every episode.

        :param agent: Agent to add
        :param player_num: the player number that this new agent can be
        """
        self.partners[self._get_partner_num(player_num)].append(agent)

    def set_partnerid(self, agent_id: int, player_num: int = 1) -> None:
        """
        Set the current partner agent to use

        :param agent_id: agent_id to use as current partner
        """
        partner_num = self._get_partner_num(player_num)
        assert(agent_id >= 0 and agent_id < len(self.partners[partner_num]))
        self.partnerids[partner_num] = agent_id

    def resample_partner(self) -> None:
        """ Resample the partner agent used """
        self.partnerids = [np.random.randint(len(plist))
                           for plist in self.partners]

    def _get_actions(self, players, obs, ego_act=None):
        print('get actions', players, self.ego_ind)
        actions = []
        for player, ob in zip(players, obs):
            if player == self.ego_ind:
                actions.append(ego_act)
            else:
                p = self._get_partner_num(player)
                agent = self.partners[p][self.partnerids[p]]
                actions.append(agent.get_action(ob))
                if not self.should_update[p]:
                    agent.update(self.total_rews[player], False)
                self.should_update[p] = True
        return np.array(actions)

    def _update_players(self, rews, done):
        for i in range(self.n_units - 1):
            nextrew = rews[i + (0 if i < self.ego_ind else 1)]
            if self.should_update[i]:
                self.partners[i][self.partnerids[i]].update(nextrew, done)

        for i in range(self.n_units):
            self.total_rews[i] += rews[i]

    def step(self, action: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool, Dict]:
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the ego_step function and the alt_step function to get to the
        next observation of the ego agent.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            done: Whether the episode has ended (need to call reset() if True)
            info: Extra information about the environment
        """
        ego_rew = 0.0

        while True:
            acts = self._get_actions(self._players, self._obs, action)
            self._players, self._obs, rews, done, info = self.n_step(acts)
            info['_partnerid'] = self.partnerids

            self._update_players(rews, done)

            ego_rew += rews[self.ego_ind] if self.ego_moved \
                else self.total_rews[self.ego_ind]

            self.ego_moved = True

            if done:
                return self._old_ego_obs, ego_rew, done, info

            if self.ego_ind in self._players:
                break

        ego_obs = self._obs[self._players.index(self.ego_ind)]
        self._old_ego_obs = ego_obs
        print('step', ego_rew, done)
        return ego_obs, ego_rew, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self.resample_partner()
        self._players, self._obs = self.n_reset()
        self.should_update = [False] * (self.n_units - 1)
        self.total_rews = [0] * self.n_units
        self.ego_moved = False

        while self.ego_ind not in self._players:
            acts = self._get_actions(self._players, self._obs)
            self._players, self._obs, rews, done, _ = self.n_step(acts)

            if done:
                raise PlayerException("Game ended before ego moved")

            self._update_players(rews, done)

        ego_obs = self._obs[self._players.index(self.ego_ind)]

        assert ego_obs is not None
        self._old_ego_obs = ego_obs
        print('reset obs', ego_obs)
        return ego_obs

    @abstractmethod
    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        """
        Perform the actions specified by the agents that will move. This
        function returns a tuple of (next agents, observations, both rewards,
        done, info).

        This function is called by the `step` function.

        :param actions: List of action provided agents that are acting on this
        step.

        :returns:
            agents: Tuple representing the agents to call for the next actions
            observations: Tuple representing the next observations (ego, alt)
            rewards: Tuple representing the rewards of all agents
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        """
        Reset the environment and return which agents will move first along
        with their initial observations.

        This function is called by the `reset` function.

        :returns:
            agents: Tuple representing the agents that will move first
            observations: Tuple representing the observations of both agents
        """



class SimultaneousEnv(MultiAgentEnv, ABC):
    """
    Base class for all 2-player simultaneous games.

    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self, partners: Optional[List] = None):
        partners = [partners] if partners else None
        super(SimultaneousEnv, self).__init__(
            ego_ind=0, n_units=2, partners=partners)

    def n_step(self, actions: List[np.ndarray]) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        return ((0, 1),) + self.multi_step(actions[0], actions[1])

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        return (0, 1), self.multi_reset()

    @abstractmethod
    def multi_step(
                    self,
                    ego_action: np.ndarray,
                    alt_action: np.ndarray
                ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                           Tuple[float, float], bool, Dict]:
        """
        Perform the ego-agent's and partner's actions. This function returns a
        tuple of (observations, both rewards, done, info).

        This function is called by the `step` function.

        :param ego_action: An action provided by the ego-agent.
        :param alt_action: An action provided by the partner.

        :returns:
            observations: Tuple representing the next observations (ego, alt)
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment and give the observation of both agents.

        This function is called by the `reset` function.

        :returns: The observations of both agents
        """


class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, config, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        DEFAULT_ENV_PARAMS = {
            "horizon": 400
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.name = config['env_name'].split('_', 1)[-1].replace('-', '_')
        self.mdp = OvercookedGridworld.from_layout_name(layout_name=self.name, rew_shaping_params=rew_shaping_params)
        mlp = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        self.base_env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        print('OvercookedMultiEnv observation space', self.observation_space)
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete(self.lA)
        print('OvercookedMultiEnv action space', self.action_space)
        self.ego_agent_idx = config['ego_agent_idx']
        self.multi_reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        print('OvercookedMultiEnv setup observation space', self.featurize_fn(dummy_state)[0], obs_shape)
        print('OvercookedMultiEnv setup observation space', self.mdp.cook_time, self.mdp.max_num_ingredients)
        high = np.ones(obs_shape, dtype=np.float32) * max(self.mdp.cook_time or 0, self.mdp.max_num_ingredients, 5)

        return gym.spaces.Box(high * 0, high, dtype=np.float32)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info['shaped_r']
        reward = reward + rew_shape

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs), (reward, reward), done, {}#info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        pass
