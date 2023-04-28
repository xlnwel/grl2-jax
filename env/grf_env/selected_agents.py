import numpy as np
import gym
import env.football.gfootball.env as football_env


class Representation:
    RAW='raw'
    CUSTOM='custom'
    MAT='mat'
    SIMPLE115='simple115v2'

class SelectedAgents(gym.Wrapper):
    def __init__(
        self, 
        env_name,
        representation=Representation.SIMPLE115,
        rewards='scoring,checkpoints',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=1000,
        logdir='results/grf',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
    ):
        if env_name == 'academy_3_vs_1_with_keeper':
            self.controlled_players = [1, 2, 3]
            n_left_agents = 4
        elif env_name == 'academy_corner':
            self.controlled_players = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_left_agents = 11
        elif env_name == 'academy_counterattack_hard':
            self.controlled_players = [5, 6, 7, 8]
            n_left_agents = 11
        elif env_name == 'academy_custom_counterattack_hard':
            self.controlled_players = [1, 2, 3, 4]
            n_left_agents = 5
        else:
            self.controlled_players = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_left_agents = 11

        other_config_options = {'action_set':'v2'}
        self.env = football_env.create_environment(
            env_name, 
            representation=representation,
            rewards=rewards,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir=logdir,
            extra_players=extra_players,
            number_of_left_players_agent_controls=n_left_agents,
            number_of_right_players_agent_controls=number_of_right_players_agent_controls,
            other_config_options=other_config_options, 
        )
        super().__init__(self.env)

        self.number_of_left_players_agent_controls = number_of_left_players_agent_controls
        self.action_dim = 20
        self.n_left_agents = n_left_agents

    def random_action(self):
        actions = []
        for i in range(self.n_left_agents):
            if i in self.controlled_players:
                actions.append(np.random.randint(self.action_dim))
            else:
                actions.append(19)
        return actions

    def reset(self):
        obs = self.env.reset()
        obs = self.get_controlled_players_data(obs)
        return obs

    def step(self, action):
        assert len(action) == self.number_of_left_players_agent_controls, (action)
        actions = []
        cid = 0
        for i in range(self.n_left_agents):
            if i in self.controlled_players:
                actions.append(action[cid])
                cid += 1
            else:
                actions.append(19)
        obs, reward, done, info = super().step(actions)
        obs = self.get_controlled_players_data(obs)
        reward = self.get_controlled_players_data(reward)

        return obs, reward, done, info

    def get_controlled_players_data(self, data):
        data = np.array([d for i, d in enumerate(data) if i in self.controlled_players])
        return data
