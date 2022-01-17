from abc import ABC, abstractmethod

class Buffer(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError

