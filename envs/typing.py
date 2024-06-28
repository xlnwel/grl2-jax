from core.typing import namedarraytuple


# for multi-processing efficiency, we do not return info at every step
EnvOutput = namedarraytuple('EnvOutput', 'obs reward discount reset')
RSSMEnvOutput = namedarraytuple('RSSMEnvOutput', 'obs reward discount reset prev_action')
# Output format of gym
GymOutput = namedarraytuple('GymOutput', 'obs reward discount')
