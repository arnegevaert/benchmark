from captum import attr

class IntegratedGradients:
    def __init__(self, forward_func, internal_batch_size=None):
        self.method = attr.IntegratedGradients(forward_func)
        self.internal_batch_size = internal_batch_size

    def __call__(self, x, target):
        return self.method.attribute(x, target=target, internal_batch_size=self.internal_batch_size)
