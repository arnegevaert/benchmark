from captum import attr


class SmoothGrad:
    def __init__(self, model, num_samples, stdev):
        self.method = attr.NoiseTunnel(attr.Saliency(model))
        self.num_samples = num_samples
        self.stdev = stdev

    def __call__(self, x, target):
        return self.method.attribute(x, target=target, nt_type="smoothgrad",
                                     n_samples=self.num_samples, stdevs=self.stdev)
