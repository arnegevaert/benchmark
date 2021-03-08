from attrbench.lib.masking import Masker
from attrbench.metrics import Metric
from typing import Callable, List
import torch
import numpy as np
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader


class _ImpactScoreDataset(Dataset):
    def __init__(self, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        attrs = attrs.reshape(attrs.shape[0], -1)
        self.sorted_indices = np.argsort(attrs)
        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples


def impact_score(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.tensor, num_steps: int,
                 strict: bool, masker: Masker, tau: float = None, writer=None):
    if not (strict or tau):
        raise ValueError("Provide value for tau when calculating non-strict impact score")
    counts = []
    ds = _ImpactScoreDataset(num_steps, samples.cpu().numpy(), attrs, masker)
    dl = DataLoader(ds, shuffle=False, batch_size=1, num_workers=4, pin_memory=True)
    with torch.no_grad():
        orig_out = model(samples)
    batch_size = samples.size(0)
    orig_confidence = softmax(orig_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
    device = samples.device

    for i, masked_samples in enumerate(dl):
        masked_samples = masked_samples[0].to(device).float()
        with torch.no_grad():
            masked_out = model(masked_samples)
        if writer:
            writer.add_images('masked samples', masked_samples, global_step=i)
        confidence = softmax(masked_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
        flipped = torch.argmax(masked_out, dim=1) != labels
        if not strict:
            flipped = flipped | (confidence <= orig_confidence * tau).squeeze()
        counts.append(flipped.sum().item())
    # [len(mask_range)]
    result = torch.tensor(counts)
    return result, batch_size


class ImpactScore(Metric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, strict: bool,
                 masker: Masker, tau: float = None, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.strict = strict
        self.masker = masker
        self.tau = tau

    def _run_single_method(self, samples, labels, attrs, writer=None):
        return impact_score(samples, labels, self.model, attrs, self.num_steps,
                            self.strict, self.masker, self.tau,
                            writer=writer)

    def get_results(self):
        result = {}
        shape = None
        for method_name in self.results:
            flipped = torch.stack([item[0] for item in self.results[method_name]], dim=0).float()
            totals = torch.tensor([item[1] for item in self.results[method_name]]).reshape(-1, 1).float()
            ratios = flipped / totals
            result[method_name] = ratios.mean(dim=0).numpy().reshape(1, -1)
            if shape is None:
                shape = result[method_name].shape
            elif result[method_name].shape != shape:
                raise ValueError(f"Inconsistent shapes for results: "
                                 f"{method_name} had {result[method_name].shape} instead of {shape}")
        return result, shape
