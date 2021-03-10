import torch
import h5py
from typing import Callable, List, Union, Tuple
from skimage.segmentation import slic
import numpy as np
from torch.utils.data import Dataset, DataLoader
from attrbench.metrics import Metric, MetricResult
from attrbench.lib import AttributionWriter
from attrbench.lib.util import corrcoef


class _PerturbationDataset(Dataset):
    def __init__(self, samples: np.ndarray, perturbation_size, num_perturbations):
        self.samples = samples
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations

    def __len__(self):
        return self.num_perturbations

    def __getitem__(self, item):
        raise NotImplementedError


class _GaussianPerturbation(_PerturbationDataset):
    # perturbation_size is stdev of noise
    def __getitem__(self, item):
        rng = np.random.default_rng(item)  # Unique seed for each item ensures no duplicate indices
        perturbation_vector = rng.normal(0, self.perturbation_size, self.samples.shape)
        perturbed_samples = self.samples - perturbation_vector
        return perturbed_samples, perturbation_vector


class _SquareRemovalPerturbation(_PerturbationDataset):
    # perturbation_size is (square height)/(image height)
    def __getitem__(self, item):
        rng = np.random.default_rng(item)  # Unique seed for each item ensures no duplicate indices
        height = self.samples.shape[2]
        width = self.samples.shape[3]
        square_size_int = int(self.perturbation_size * height)
        x_loc = rng.integers(0, width - square_size_int, size=1).item()
        y_loc = rng.integers(0, height - square_size_int, size=1).item()
        perturbation_mask = np.zeros(self.samples.shape)
        perturbation_mask[:, :, x_loc:x_loc + square_size_int, y_loc:y_loc + square_size_int] = 1
        perturbation_vector = self.samples * perturbation_mask
        perturbed_samples = self.samples - perturbation_vector
        return perturbed_samples, perturbation_vector


class _SegmentRemovalPerturbation(_PerturbationDataset):
    # perturbation size is number of segments
    def __init__(self, samples, perturbation_size, num_perturbations):
        super().__init__(samples, perturbation_size, num_perturbations)
        seg_samples = np.stack([slic(np.transpose(samples[i, ...], (1, 2, 0)),
                                     start_label=0, slic_zero=True)
                                for i in range(samples.shape[0])])
        self.seg_samples = np.expand_dims(seg_samples, axis=1)

    def __getitem__(self, item):
        rng = np.random.default_rng(item)  # Unique seed for each item ensures no duplicate indices
        perturbed_samples, perturbation_vectors = [], []
        # This needs to happen per sample, since samples don't necessarily have
        # the same number of segments
        for i in range(self.samples.shape[0]):
            seg_sample = self.seg_samples[i, ...]
            sample = self.samples[i, ...]
            # Get all segment numbers
            all_segments = np.unique(seg_sample)
            # Select segments to mask
            segments_to_mask = rng.choice(all_segments, self.perturbation_size, replace=False)
            # Create boolean mask of pixels that need to be removed
            to_remove = np.isin(seg_sample, segments_to_mask)
            # Create perturbation vector by multiplying mask with image
            perturbation_vector = sample * to_remove.astype(np.float)
            perturbed_samples.append((sample - perturbation_vector).astype(np.float))
            perturbation_vectors.append(perturbation_vector)
        return np.stack(perturbed_samples, axis=0), np.stack(perturbation_vectors, axis=0)


_PERTURBATION_CLASSES = {
    "gaussian": _GaussianPerturbation,
    "square": _SquareRemovalPerturbation,
    "segment": _SegmentRemovalPerturbation
}


_OUT_FNS = {
    "mse": lambda a, b: ((a - b) ** 2).mean(dim=1, keepdim=True),
    "corr": lambda a, b: torch.tensor(corrcoef(a.numpy(), b.numpy()))
}


def _compute_perturbations(samples: torch.Tensor, labels: torch.Tensor, model: Callable, perturbation_mode: str,
                           perturbation_size: float, num_perturbations: int, writer: AttributionWriter = None):
    device = samples.device
    if perturbation_mode not in _PERTURBATION_CLASSES.keys():
        raise ValueError(f"Invalid perturbation mode {perturbation_mode}. "
                         f"Valid options are {', '.join(list(_PERTURBATION_CLASSES.keys()))}")
    perturbation_ds = _PERTURBATION_CLASSES[perturbation_mode](samples.cpu().numpy(),
                                                               perturbation_size, num_perturbations)
    perturbation_dl = DataLoader(perturbation_ds, batch_size=1, num_workers=4, pin_memory=True)

    # Get original model output
    with torch.no_grad():
        orig_output = (model(samples)).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]

    pert_vectors = []
    pred_diffs = []
    for i_pert, (perturbed_samples, perturbation_vector) in enumerate(perturbation_dl):
        # Get perturbation vector I and perturbed samples (x - I)
        perturbed_samples = perturbed_samples[0].float().to(device)
        perturbation_vector = perturbation_vector[0].float()
        if writer:
            writer.add_images("perturbation_vector", perturbation_vector, global_step=i_pert)
            writer.add_images("perturbed_samples", perturbed_samples, global_step=i_pert)

        # Get output of model on perturbed sample
        with torch.no_grad():
            perturbed_output = model(perturbed_samples).gather(dim=1, index=labels.unsqueeze(-1))
        # Save the prediction difference and perturbation vector
        pred_diffs.append(orig_output - perturbed_output)  # [batch_size, 1]
        pert_vectors.append(perturbation_vector)  # [batch_size, *sample_shape]
    pert_vectors = torch.stack(pert_vectors, dim=1)  # [batch_size, num_perturbations, *sample_shape]
    pred_diffs = torch.cat(pred_diffs, dim=1).cpu()  # [batch_size, num_perturbations]
    return pert_vectors, pred_diffs


def _compute_result(pert_vectors: torch.Tensor, pred_diffs: torch.Tensor, attrs: np.ndarray,
                    out_fn: Tuple[str]):
    # Replicate attributions along channel dimension if necessary (if explanation has fewer channels than image)
    attrs = torch.tensor(attrs).float()
    if attrs.shape[1] != pert_vectors.shape[-3]:
        shape = [1 for _ in range(len(attrs.shape))]
        shape[1] = pert_vectors.shape[-3]
        attrs = attrs.repeat(*tuple(shape))

    # Calculate dot product between each sample and its corresponding perturbation vector
    # This is equivalent to diagonal of matmul
    attrs = attrs.flatten(1).unsqueeze(1)  # [batch_size, 1, -1]
    pert_vectors = pert_vectors.flatten(2)  # [batch_size, num_perturbations, -1]
    dot_product = (attrs * pert_vectors).sum(dim=-1)  # [batch_size, num_perturbations]

    result = []
    for fn in out_fn:
        result.append(_OUT_FNS[fn](dot_product, pred_diffs))
    return tuple(result)


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
               perturbation_mode: str, perturbation_size: float, num_perturbations: int,
               out_fn: Union[Tuple[str], str] = "mse",
               writer: AttributionWriter = None):
    pert_vectors, pred_diffs = _compute_perturbations(samples, labels, model, perturbation_mode,
                                                      perturbation_size, num_perturbations, writer)
    if type(out_fn) == str:
        out_fn = (out_fn,)
    return _compute_result(pert_vectors, pred_diffs, attrs, out_fn)


class Infidelity(Metric):
    def __init__(self, model: Callable, method_names: List[str], perturbation_mode: str,
                 perturbation_size: float, num_perturbations: int, out_fn: Union[Tuple[str], str] = "mse",
                 writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations
        self.out_fn = (out_fn,) if type(out_fn) == str else out_fn
        if self.writer_dir is not None:
            self.writers["general"] = AttributionWriter(self.writer_dir)
        self.result = InfidelityResult(method_names, perturbation_mode, perturbation_size, self.out_fn)

    def run_batch(self, samples, labels, attrs_dict: dict):
        # First calculate perturbation vectors and predictions differences, these can be re-used for all methods
        writer = self.writers["general"] if self.writers is not None else None
        pert_vectors, pred_diffs = _compute_perturbations(samples, labels, self.model,
                                                          self.perturbation_mode, self.perturbation_size,
                                                          self.num_perturbations, writer)
        for method_name in attrs_dict:
            self.result.append(method_name, _compute_result(pert_vectors, pred_diffs, attrs_dict[method_name],
                                                            self.out_fn))


class InfidelityResult(MetricResult):
    def __init__(self, method_names: List[str], perturbation_mode: str, perturbation_size: float,
                 out_fn: Tuple[str]):
        super().__init__(method_names)
        self.inverted = True
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size
        self.out_fn = out_fn
        self.data = {
            m_name: {fn: [] for fn in out_fn} for m_name in method_names
        }

    def append(self, method_name, batch):
        batch = tuple(batch)
        if len(self.out_fn) != len(batch):
            raise ValueError(f"Invalid number of results: expected {len(self.out_fn)}, got {len(batch)}.")
        for i, fn in enumerate(self.out_fn):
            self.data[method_name][fn].append(batch[i])

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            method_group = group.create_group(method_name)
            for fn in self.out_fn:
                if type(self.data[method_name][fn]) == list:
                    method_group.create_dataset(fn, data=torch.cat(self.data[method_name][fn]).numpy())
                else:
                    method_group.create_dataset(fn, data=self.data[method_name][fn])
        group.attrs["perturbation_mode"] = self.perturbation_mode
        group.attrs["perturbation_size"] = self.perturbation_size

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        out_fn = tuple(group[method_names[0]].keys())
        result = cls(method_names, group.attrs["perturbation_mode"], group.attrs["perturbation_size"], out_fn)
        result.data = {
            m_name: {fn: np.array(group[m_name][fn]) for fn in out_fn} for m_name in method_names
        }
        return result

