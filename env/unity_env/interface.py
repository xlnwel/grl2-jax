import os

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class UnityInterface:
    def __init__(
        self,
        worker_id=0,
        file_name=None,
        port=5005,
        render=False,
        seed=42,
        timeout_wait=10,
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
        self._default_reset_signals = ['9'] + ['0'] * self._n_envs
        self._real_done = real_done

        self._side_channels = self._initialize_all_side_channels(initialize_config, engine_config)
        print('worker_id', worker_id)
        env_kwargs = dict(
            seed=seed,
            worker_id=worker_id,
            timeout_wait=timeout_wait,
            side_channels=list(self._side_channels.values())
        )

        if file_name is not None:
            file_name = os.path.join(os.path.abspath('.'), file_name)
            env_kwargs.update(
                file_name=file_name,
                base_port=port,
                no_graphics=not render
            )
        self.env = UnityEnvironment(**env_kwargs)
        self.env.reset()
        self._behavior_names = list(self.env.behavior_specs.keys())
        self._behavior_specs = list(self.env.behavior_specs.values())

    def _initialize_all_side_channels(self, initialize_config, engine_config):
        engine_configuration_channel = EngineConfigurationChannel()
        engine_configuration_channel.set_configuration_parameters(**engine_config)
        float_properties_channel = EnvironmentParametersChannel()
        float_properties_channel.set_float_parameter('n_envs', self._n_envs)
        for k, v in initialize_config.items():
            float_properties_channel.set_float_parameter(k, v)
        return dict(engine_configuration_channel=engine_configuration_channel,
                    float_properties_channel=float_properties_channel)

    def get_behavior_names(self):
        return self._behavior_names

    def get_behavior_specs(self):
        return self._behavior_specs

    def reset(self):
        self.env.reset()
        return self.get_obs()
    
    def reset_envs_with_ids(self, eids: list):
        """ Reset environments specified by the given ids """
        signal = self._default_reset_signals.copy()
        for i in eids:
            signal[i+1] = '1'
        self._side_channels['float_properties_channel'].set_float_parameter('resets', float(''.join(signal)))

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
        return self.get_obs()

    def get_obs(self):
        decision_steps = {}
        terminal_steps = {}
        for bn in self._behavior_names:
            ds, ts = self.env.get_steps(bn)
            decision_steps[bn] = ds
            terminal_steps[bn] = ts
        return decision_steps, terminal_steps
