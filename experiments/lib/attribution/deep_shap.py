from captum import attr
from torch.utils.data import Dataset, DataLoader


class DeepShap:
    def __init__(self, model, reference_dataset: Dataset, n_baseline_samples: int):
        self.model = model
        self.reference_dataset = reference_dataset
        self.n_baseline_samples = n_baseline_samples
        self.method = attr.DeepLiftShap(model)
        self.ref_sampler = DataLoader(
            dataset=self.reference_dataset,
            batch_size=self.n_baseline_samples,
            shuffle=True, drop_last=True
        )

    def __call__(self, x, target):
        ref_batch = next(iter(self.ref_sampler))[0].to(x.device)
        return self.method.attribute(x, baselines=ref_batch, target=target)
