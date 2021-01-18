from captum import attr


class SmoothGrad:
    def __init__(self, forward_func):
        self.method = attr.NoiseTunnel(attr.Saliency(forward_func))

    def __call__(self, x, target):
        return self.method.attribute(x, target=target, nt_type="smoothgrad", n_samples=50, stdev=0.15)
