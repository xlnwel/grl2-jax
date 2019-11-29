from replay.uniform import UniformReplay
from replay.per import ProportionalPER


def create_replay(config, state_shape, state_dtype, 
                action_dim, action_dtype, gamma, 
                has_next_state=False):
    buffer_type = config['type']
    if buffer_type == 'uniform': 
        return UniformReplay(
            config, state_shape, state_dtype, 
            action_dim, action_dtype, gamma, 
            has_next_state=has_next_state)
    elif buffer_type == 'proportional':
        return ProportionalPER(
            config, state_shape, state_dtype, 
            action_dim, action_dtype, gamma, 
            has_next_state=has_next_state)
    else:
        raise NotImplementedError()