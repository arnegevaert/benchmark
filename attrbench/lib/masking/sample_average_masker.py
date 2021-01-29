from attrbench.lib.masking import Masker
import torch


class SampleAverageMasker(Masker):
    def mask(self, samples, indices):
        res = []
        for i in range(samples.shape[0]):
            sample = samples[i, ...]
            baseline = torch.zeros(sample.shape).float()
            for c in range(sample.shape[0]):
                baseline[c, ...] = torch.mean(sample[c, ...])
            to_mask = torch.zeros(sample.shape).flatten(0 if self.feature_level == "channel" else 1)
            if self.feature_level == "channel":
                to_mask[indices[i]] = 1.
            else:
                to_mask[:, indices[i]] = 1.
            to_mask = to_mask.reshape(sample.shape)
            res.append(sample - (to_mask * sample) + (to_mask * baseline))
        return torch.stack(res, dim=0)
