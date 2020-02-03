class Model:
    def __init__(self):
        pass

    def train(self, epochs, save=True):
        raise NotImplementedError

    # Expects input to have batch dimension
    def predict(self, x):
        raise NotImplementedError


