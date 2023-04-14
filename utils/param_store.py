from collections import OrderedDict


class ParameterStore:
    def __init__(self, config):
        self.params = config

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
