import numpy as np


# TODO masker should take attributions as constructor argument and then implement the necessary
#      functions (mask_highest, keep_highest, mask_lowest, keep_lowest, mask_random, mask_all)
class Masker:
    def __init__(self,samples: np.ndarray, attributions: np.ndarray, feature_level, segmentation: np.ndarray =None):
        if feature_level not in ("channel", "pixel"):
            raise ValueError(f"feature_level must be 'channel' or 'pixel'. Found {feature_level}.")
        self.feature_level = feature_level
        self.baseline = None
        self.samples=samples
        self.attributions=attributions
        self.segments=segmentation
        self.sorted_indices = attributions.reshape(attributions.shape[0], -1).argsort()


    #### can be deleted later
    def initialize_samples(self, samples, attributions, segmentation = None):
        if not self.check_attribution_shape(samples,attributions):
            raise ValueError("Attributions are not compatible with masker")
        self.initialize_baselines(samples)
        self.samples=samples
        self.attributions=attributions
        self.segments=segmentation
        self.sorted_indices = attributions.reshape(attributions.shape[0], -1).argsort()

    # TODO: deal with segments: see irof
    def mask_top(self, k, segmented=False):
        if k==0:
            return self.samples
        return self.mask(self.samples, self.sorted_indices[:,-k:])
    def mask_bot(self, k, segmented=False):
        return self.mask(self.samples, self.sorted_indices[:,:k])
    def keep_top(self, k, segmented=False):
        if k==0:
            return self.samples
        return self.mask(self.samples, self.sorted_indices[:, :-k])
    def keep_bot(self,k, segmented=False):
        return self.mask(self.samples, self.sorted_indices[:,k:])
    def mask_rand(self,k,segmented=False):
        if k==0:
            return self.samples
        indices = np.arange(self.sorted_indices.shape(-1))
        indices = np.tile(indices, (self.sorted_indices.shape[0],1))
        rng=np.random.default_rng()
        rng.shuffle(indices,axis=1)
        return self.mask(self.samples, indices[:,:k])
    def mask_all(self):
        return self.mask(self.samples, self.sorted_indices)

    def predict_masked(self, samples, indices, model, return_masked_samples=False):
        masked = self.mask(samples, indices)
        with torch.no_grad():
            pred = model(masked)
        if return_masked_samples:
            return pred, masked
        return pred

    def check_attribution_shape(self, samples, attributions):
        if self.feature_level == "channel":
            # Attributions should be same shape as samples
            return list(samples.shape) == list(attributions.shape)
        elif self.feature_level == "pixel":
            # attributions should have the same shape as samples,
            # except the channel dimension must be 1
            aggregated_shape = list(samples.shape)
            aggregated_shape[1] = 1
            return aggregated_shape == list(attributions.shape)
    #TODO: private
    def mask(self, samples: np.ndarray, indices: np.ndarray):
        if self.baseline is None:
            raise ValueError("Masker was not initialized.")
        batch_size, num_channels, rows, cols = samples.shape
        num_indices = indices.shape[1]
        batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()

        #to_mask = torch.zeros(samples.shape).flatten(1 if self.feature_level == "channel" else 2)
        to_mask = np.zeros(samples.shape)
        if self.feature_level == "channel":
            to_mask = to_mask.reshape((to_mask.shape[0], -1))
        else:
            to_mask = to_mask.reshape((to_mask.shape[0], to_mask.shape[1], -1))
        if self.feature_level == "channel":
            to_mask[batch_dim, indices] = 1.
        else:
            try:
                to_mask[batch_dim, :, indices] = 1.
            except IndexError:
                raise ValueError("Masking index was out of bounds. "
                                 "Make sure the masking policy is compatible with method output.")
        to_mask = to_mask.reshape(samples.shape)
        return self.mask_boolean(samples, to_mask)

    def mask_boolean(self, samples, bool_mask):
        bool_mask = bool_mask
        return samples - (bool_mask * samples) + (bool_mask * self.baseline)
    # why not return samples[bool_mask] = self.baseline[bool_mask] ?


    def initialize_baselines(self, samples):
        raise NotImplementedError

