import uuid

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)


class RuntimeEnvironmentParametersChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        print(f"Unity->Python:{msg.read_string()}")

    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        print('send')
        super().queue_message_to_send(msg)


string_log = RuntimeEnvironmentParametersChannel()

env = UnityEnvironment(side_channels=[string_log])
env.reset()

i = 0
while True:
    key = "reset"
    i += 1
    env.step()
    if i % 1 == 0:
        value = '91_0_1'
        string_log.send_string('{'f"key:\"{key}\",value:\"{value}\""'}')
        continue

    if i == 1000:
        break

env.close()
