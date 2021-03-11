from typing import Callable, List

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from attrbench.lib.masking import Masker
from attrbench.metrics import Metric
from ._dataset import _ImpactScoreDataset
from .result import ImpactScoreResult


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
            writer.add_images("masked samples", masked_samples, global_step=i)
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
        self.result = ImpactScoreResult(method_names, strict, tau)

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            if method_name not in self.result.method_names:
                raise ValueError(f"Invalid method name: {method_name}")
            flipped, total = impact_score(samples, labels, self.model, attrs_dict[method_name], self.num_steps,
                                          self.strict, self.masker, self.tau, writer=self._get_writer(method_name))
            self.result.append(method_name, (flipped, total))

