from captum import attr


class SmoothGrad:
    def __init__(self, model, num_samples, stdev,internal_batch_size):
        self.method = attr.NoiseTunnel(attr.Saliency(model))
        self.num_samples = num_samples
        self.stdev = stdev
        self.internal_batch_size = internal_batch_size

    def __call__(self, x, target):
        # sigma = self.noise_level / (x.max()-x.min()) # follows paper more closely, but not perfectly.
        # in paper the sigma is set per image, here per batch
        return self.method.attribute(x, target=target, nt_type="smoothgrad",
                                     nt_samples=self.num_samples,
                                     nt_samples_batch_size=self.internal_batch_size,
                                     stdevs=self.stdev)
