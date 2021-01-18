from captum import attr


class IntegratedGradients:
    def __init__(self, model, internal_batch_size):
        self.method = attr.IntegratedGradients(model)
        self.internal_batch_size = internal_batch_size

    def __call__(self, x, target):
        return self.method.attribute(x, target=target, internal_batch_size=self.internal_batch_size)
