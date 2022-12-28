class Dummy:
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, **kwargs):
        return x

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
