class Component:
    def __init__(self, app):
        self.app = app

    def render(self):
        raise NotImplementedError
