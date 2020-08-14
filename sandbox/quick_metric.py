from os import path
from torch.utils.data import DataLoader
import itertools
from attrbench import datasets, attribution, models
from attrbench.evaluation.insertion_deletion_curves import insertion_deletion_curves

device = "cuda"
dataset = datasets.Cifar(data_location=path.join("../data", "CIFAR10"), train=False)
mask_range = list(range(30, 32*32, 30))

model = models.Resnet("resnet18", True, 10, "../data/models/CIFAR10/resnet18.pt")
model.to(device)
model.eval()

kwargs = {
    "normalize": False,  # Normalizing isn't necessary, only order of values counts
    "aggregation_fn": "avg"
}

attribution_methods = {
    "Gradient": attribution.Gradient(model, **kwargs),
    "SmoothGrad": attribution.SmoothGrad(model, **kwargs),
    "InputXGradient": attribution.InputXGradient(model, **kwargs),
    "IntegratedGradients": attribution.IntegratedGradients(model, **kwargs),
    "GuidedBackprop": attribution.GuidedBackprop(model, **kwargs),
    "Deconvolution": attribution.Deconvolution(model, **kwargs),
    #"Ablation": attribution.Ablation(model, **kwargs),
    "GuidedGradCAM": attribution.GuidedGradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs),
    "GradCAM": attribution.GradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs)
}

dataloader = itertools.islice(DataLoader(dataset, batch_size=4, num_workers=4), 2)
result = insertion_deletion_curves(dataloader, dataset.sample_shape, model,
                                   attribution_methods, mask_range, dataset.mask_value,
                                   pixel_level_mask=True, device=device,
                                   mode="insertion", output_transform="identity")
result.save_json("result.json")