from attrbench.lib.masking import Masker


class RandomMasker(Masker):
    def __init__(self, feature_level, additive=False):
        super().__init__(feature_level)
        self.additive = additive

    def mask(self, samples, indices):
        return samples

    def predict_masked(self, samples, indices, model, return_masked_samples=False):
        pass  # TODO
