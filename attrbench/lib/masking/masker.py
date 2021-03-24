import numpy as np



class Masker:
    def __init__(self, samples: np.ndarray, attributions: np.ndarray):
        if not self._check_attribution_shape(samples,attributions):
            raise ValueError(f"samples and attribution shape not compatible."
                             f"Found shapes {samples.shape} and {attributions.shape}")

        self.baseline = None
        self.samples=samples
        self.attributions=attributions
        self.sorted_indices = attributions.reshape(attributions.shape[0], -1).argsort()

    def get_total_features(self):
        return self.sorted_indices.shape[1]

    def mask_top(self, k):
        if k==0:
            return self.samples
        else:
            return self._mask(self.samples, self.sorted_indices[:, -k:])
    def mask_bot(self, k):
        return self._mask(self.samples, self.sorted_indices[:, :k])
    def keep_top(self, k):
        if k==0:
            return self.samples
        else:
            return self._mask(self.samples, self.sorted_indices[:, :-k])
    def keep_bot(self,k):
        return self._mask(self.samples, self.sorted_indices[:, k:])

    def mask_rand(self,k,return_indices=False):
        if k==0:
            return self.samples
        rng = np.random.default_rng()
        shape = self.sorted_indices.shape
        indices = np.arange(shape[-1])
        indices = np.tile(indices, (shape[0],1))
        rng.shuffle(indices,axis=1)
        indices=indices[:, :k]
        masked_samples = self._mask(self.samples, indices)
        if return_indices: return masked_samples, indices
        return masked_samples
    def mask_all(self,segmented=False):
        return self._mask(self.samples, self.sorted_indices)

    def _check_attribution_shape(self, samples, attributions):
        raise NotImplementedError

    def _mask(self, samples: np.ndarray, indices: np.ndarray):
        raise NotImplementedError

    def _mask_boolean(self, samples, bool_mask):
        raise NotImplementedError

    def initialize_baselines(self, samples):
        raise NotImplementedError

