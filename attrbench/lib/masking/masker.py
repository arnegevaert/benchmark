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
        if self.segments is not None:
            self.segmented_attrs = self.segment_attributions(self.segments,self.attributions).argsort()
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

    def segment_attributions(self,seg_images: np.ndarray, attrs: np.ndarray) -> np.ndarray:
        segments = np.unique(seg_images)
        seg_img_flat = seg_images.reshape(seg_images.shape[0], -1)
        attrs_flat = attrs.reshape(attrs.shape[0], -1)
        avg_attrs = np.zeros((seg_images.shape[0], len(segments)))
        for i, seg in enumerate(segments):  # Segments should be 0, ..., n, but we use enumerate just in case
            mask = (seg_img_flat == seg).astype(np.long)
            masked_attrs = mask * attrs_flat
            mask_size = np.sum(mask, axis=1)
            sum_attrs = np.sum(masked_attrs, axis=1)
            mean_attrs = np.divide(sum_attrs, mask_size, out=np.zeros_like(sum_attrs), where=mask_size!=0)
            # If seg does not exist for image, mean_attrs will be nan. Replace with -inf.
            avg_attrs[:, i] = np.nan_to_num(mean_attrs, nan=-np.inf)
        return avg_attrs

    # TODO: deal with segments: see irof
    def mask_top(self, k, segmented=False):
        if k==0:
            return self.samples
        if segmented:
            return self.mask_segments(self.samples,self.segments,self.segmented_attrs[:,-k:])
        else:
            return self.mask(self.samples, self.sorted_indices[:,-k:])
    def mask_bot(self, k, segmented=False):
        if segmented:
            return self.mask_segments(self.samples,self.segments,self.segmented_attrs[:,:k])
        else:
            return self.mask(self.samples, self.sorted_indices[:,:k])
    def keep_top(self, k, segmented=False):
        if k==0:
            return self.samples
        if segmented:
            return self.mask_segments(self.samples,self.segments,self.segmented_attrs[:,:-k])
        else:
            return self.mask(self.samples, self.sorted_indices[:, :-k])
    def keep_bot(self,k, segmented=False):
        if segmented:
            return self.mask_segments(self.samples,self.segments,self.segmented_attrs[:,k:])
        else:
            return self.mask(self.samples, self.sorted_indices[:,k:])

    def mask_rand(self,k,segmented=False):
        if k==0:
            return self.samples
        indices = self.segmented_attrs if segmented else self.sorted_indices
        shape = indices.shape
        indices = np.arange(shape(-1))
        indices = np.tile(indices, (shape[0],1))
        rng=np.random.default_rng()
        rng.shuffle(indices,axis=1)
        if segmented:
            return self.mask_segments(self.samples,self.segments,indices[:,:k])
        else:
            return self.mask(self.samples, indices[:,:k])
    def mask_all(self,segmented=False):
        if segmented:
            return self.mask_segments(self.samples,self.segments,self.segmented_attrs)
        else:
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

    def mask_segments(self,images: np.ndarray, seg_images: np.ndarray, segments: np.ndarray) -> np.ndarray:
        if not (images.shape[0] == seg_images.shape[0] and images.shape[0] == segments.shape[0] and
                images.shape[-2:] == seg_images.shape[-2:]):
            raise ValueError(f"Incompatible shapes: {images.shape}, {seg_images.shape}, {segments.shape}")
        bool_masks = []
        for i in range(images.shape[0]):
            seg_img = seg_images[i, ...]
            segs = segments[i, ...]
            bool_masks.append(np.isin(seg_img, segs))
        bool_masks = np.stack(bool_masks, axis=0)
        return self.mask_boolean(images, bool_masks)

    def mask_boolean(self, samples, bool_mask):
        bool_mask = bool_mask
        return samples - (bool_mask * samples) + (bool_mask * self.baseline)
    # why not return samples[bool_mask] = self.baseline[bool_mask] ?


    def initialize_baselines(self, samples):
        raise NotImplementedError

