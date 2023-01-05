from abc import abstractmethod
from typing import Dict
from torch import nn


class MethodFactory:
    @abstractmethod
    def __call__(self, model: nn.Module) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_method_names(self):
        raise NotImplementedError
