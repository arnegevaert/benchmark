from methods import Method
from models import Model


class Metric:
    def __init__(self, model: Model, method: Method):
        self.model = model
        self.method = method
