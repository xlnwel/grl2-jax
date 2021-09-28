from utility.typing import namedarraytuple


# for multi-processing efficiency, we do not return info at every step
EnvOutput = namedarraytuple('EnvOutput', 'obs reward discount reset')
# Output format of gym
GymOutput = namedarraytuple('GymOutput', 'obs reward discount')
