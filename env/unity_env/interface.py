import os
import uuid
from copy import deepcopy

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np


class UnityInterface:
    def __init__(
        self,
        worker_id=0,
        file_name=None,
        port=5005,
        render=False,
        seed=42,
        timeout_wait=1000,
        n_envs=1,
        real_done=False,
        initialize_config={},
        engine_config={
            'width': 84,
            'height': 84,
            'quality_level': 5,
            'time_scale': 20,
            'target_frame_rate': -1,
            'capture_frame_rate': 60
        },
        **kwargs
    ):
        """ This class encapsulates operations that interact with a Unity environment.
        Ideally, you should not expect to see any classes from mlagents outside this class

        :param worker_id: Client id; it will added to port. Increase it when creating multiple Unity instances
        :param file_name: The relative executable file path; it will connect to the Unity editor if it's None
        :param port: Unity port，The default port for the Unity editor is 5004 and that for a Unity executable file is 5005
        :param render: If render the environment. It's only effective for executable filess
        :param seed: Random seed
        :param timeout_wait: Maximum wait time for connecting to the environment
        :param n_envs: The number of environment running in parallel, same as the number of environment copies specified when compling the Unity environment
        :param real_done: Condition for done. If real_done=False, exceeding the maximum number of steps is taken as done
        :param initialize_config: Initialization configuration
        :param engine_config: Engine configuration
        :param kwargs:
        """
        self._n_envs = n_envs
        self._real_done = real_done
        e_b = self._initialize_all_side_channels(engine_config)
        self._side_channel = RuntimeEnvironmentParametersChannel()
        env_kwargs = dict(
            seed=seed,
            worker_id=worker_id,
            timeout_wait=timeout_wait,
            side_channels=[self._side_channel, e_b]
        )

        print('filename', file_name)
        if file_name is not None:
            if not file_name.startswith('/'):
                file_name = os.path.join(os.path.abspath('.'), file_name)
            env_kwargs.update(
                file_name=file_name,
                base_port=port,
                no_graphics=not render
            )
        print(env_kwargs)
        self.env = UnityEnvironment(**env_kwargs)
        self.env.reset()
        self._behavior_names = list(self.env.behavior_specs.keys())
        self._behavior_specs = list(self.env.behavior_specs.values())
        self.last_reset = -1  # 0 means reset by id

    def get_behavior_names(self):
        return self._behavior_names

    def _initialize_all_side_channels(self, engine_config):
        engine_configuration_channel = EngineConfigurationChannel()
        engine_configuration_channel.set_configuration_parameters(**engine_config)

        return engine_configuration_channel

    def get_behavior_specs(self):
        return self._behavior_specs

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def reset_envs_with_ids(self, eids: list):
        """ Reset environments specified by the given ids """
        key = 'reset'
        for i in eids:
            self._side_channel.send_string('{'f"key: \"{key}\",value:{i}"'}')
        self.last_reset = 0

    def get_action_tuple(self) -> ActionTuple:
        """ Get an action_tuple instance"""
        return ActionTuple()

    def set_actions(self, name: str, action_tuple: ActionTuple):
        """ Set actions before taking an environment step
        Do not modify this class. Instead, call <get_action_tuple> to obtain
        an action_tuple first.
        """
        # reference: https://github.com/Unity-Technologies/ml-agents/blob/d34f3cd6ee078782b22341e4ceb958359069ab60/ml-agents-envs/mlagents_envs/tests/test_set_action.py
        self.env.set_actions(name, action_tuple)

    def step(self):
        self.env.step()

        ds, ts = self.get_obs()

        # 如果有环境done了，则空step一步
        #loop_flag = True
        last_ts = None
        #while loop_flag:
        #loop_flag = False
        for k, v in ts.items():
            if len(v) != 0:
                 #loop_flag = True
                 last_ts = deepcopy(ts)
                 self.env.step()
                 ds, ts = self.get_obs()
                 break
        ts = last_ts if last_ts is not None else ts
        if last_ts is not None:
            return True, ds, ts
        else:
            return False, ds, ts

    def get_obs(self):
        decision_steps = {}
        terminal_steps = {}
        for bn in self._behavior_names:
            ds, ts = self.env.get_steps(bn)
            decision_steps[bn] = ds
            terminal_steps[bn] = ts
        return decision_steps, terminal_steps


class RuntimeEnvironmentParametersChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        print(f"Unity->Python:{msg.read_string()}")

    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)


if __name__ == '__main__':
    unity_config = {}
    unity_config['n_envs'] = 1
    # unity_config['file_name'] = 'E:\FlightCombat\FightSimulator\FightSimulator\Packages\\test_view\T2.exe'
    env = UnityInterface(**unity_config)
    names = ['blue_main?team=0', 'blue_sup_0?team=0', 'red_main?team=0', 'red_sup_0?team=0', 'red_sup_1?team=0',
             'red_sup_2?team=0', 'red_sup_3?team=0'
             # 'blue_main?team=1', 'blue_sup_0?team=1', 'red_main?team=1', 'red_sup_0?team=1', 'red_sup_1?team=1', 'red_sup_2?team=1', 'red_sup_3?team=1'
             ]
    for i in range(1, 1000):
        env.reset_envs_with_ids([1])
        # for name in names:
        #     action_tuple = env.get_action_tuple()
        #     if name.startswith('red_main'):
        #         action_tuple.add_discrete(np.array([[1, 1, 1]]))
        #     else:
        #         action_tuple.add_discrete(np.array([[1, 1]]))
        #
        #     action_tuple.add_continuous(np.array([[0, 0]]))
        #     env.set_actions(name, action_tuple)
        ds, ts = env.step()
        # print('env0:{}, env2:{}'.format(ds['blue_main?team=0'].obs[1][0][2:4], ds['blue_main?team=1'].obs[1][0][2:4]))
