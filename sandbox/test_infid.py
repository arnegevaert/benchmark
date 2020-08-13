from torch.utils.data import DataLoader
import itertools
from attrbench import datasets, attribution, models
from attrbench.evaluation.infidelity import infidelity

device = "cuda"
dataset = datasets.MNIST("../data/MNIST", train=False)
model = models.BasicCNN(True, 10, "../data/models/MNIST/cnn.pt")
model.to(device)
model.eval()

kwargs = {
    "normalize": True,
    "aggregation_fn": "avg"
}

attribution_methods = {
    "Gradient": attribution.Gradient(model, **kwargs),
    #"SmoothGrad": attribution.SmoothGrad(model, **kwargs),
    "InputXGradient": attribution.InputXGradient(model, **kwargs),
    "IntegratedGradients": attribution.IntegratedGradients(model, **kwargs),
    #"GuidedBackprop": attribution.GuidedBackprop(model, **kwargs),
    #"Deconvolution": attribution.Deconvolution(model, **kwargs),
    #"Ablation": attribution.Ablation(model, **kwargs),
    #"GuidedGradCAM": attribution.GuidedGradCAM(model, model.get_last_conv_layer(), **kwargs),
    #"GradCAM": attribution.GradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs)
}

dataloader = itertools.islice(DataLoader(dataset, batch_size=64), 16)
result = infidelity(dataloader, model, attribution_methods, [0.1, 0.2, 0.3], 16, True, device, "identity")
fig, ax = result.plot(ci=True)
fig.show()
