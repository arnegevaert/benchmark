from captum import attr

class DeepLift:
    def __init__(self, model):
        self.method = attr.DeepLift(model)


    def __call__(self, x, target):
        self.method.attribute(x, target=target)