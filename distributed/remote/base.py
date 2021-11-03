import ray


class RayBase:
    def register_handler(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @classmethod
    def as_remote(cls, **kwargs):
        return ray.remote(**kwargs)(cls)
