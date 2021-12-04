import ray


from core.tf_config import configure_gpu, silence_tf_logs


class RayBase:
    def __init__(self):
        silence_tf_logs()
        configure_gpu()

    def register_handler(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @classmethod
    def as_remote(cls, **kwargs):
        if kwargs:
            return ray.remote(**kwargs)(cls)
        return ray.remote(cls)
