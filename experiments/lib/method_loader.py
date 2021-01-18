import yaml
from .attribution import *


class MethodLoader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_config(self, loc):
        with open(loc) as fp:
            data = yaml.full_load(fp)
            print(data)
