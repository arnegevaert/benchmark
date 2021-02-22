from experiments.general_imaging.dataset_models import get_dataset_model
from experiments.lib import attribution
from attrbench.functional import iiof
from attrbench.lib.masking import ConstantMasker
from torch.utils.data import DataLoader


if __name__ == "__main__":
    ds, model, patch_folder = get_dataset_model("ImageNette")
    dl = DataLoader(ds, batch_size=32)
    model.eval()
    model.to("cuda")

    method = attribution.PixelAggregation(attribution.InputXGradient(model), "avg")

    samples, labels = next(iter(dl))
    samples = samples.to("cuda")
    labels = labels.to("cuda")
    attrs = method(samples, target=labels)
    masker = ConstantMasker(feature_level="pixel")

    res = iiof(samples, labels, model, attrs, masker)
