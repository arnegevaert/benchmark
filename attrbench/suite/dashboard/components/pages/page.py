from attrbench.suite.dashboard.components import Component


class Page(Component):
    def __init__(self, result_obj):
        self.result_obj = result_obj

    def render(self):
        raise NotImplementedError()
