
from core.decorator import config


class ParameterServer:
    @config
    def __init__(self) -> None:
        self._agents = None
    
    def add_strategy(self, aid, sid, strategy):
        self._agents[aid].add_strategy(sid, strategy)
        
    def save(self):
        for a in self._agents:
            a.save()

    def restore(self):
        for a in self._agents:
            a.restore()
