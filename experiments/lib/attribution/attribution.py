import torch
from skimage.filters import sobel


def normalize_attributions(attrs):
    abs_attrs = torch.abs(attrs.flatten(1))
    max_abs_attr_per_image = torch.max(abs_attrs, dim=1)[0]
    if torch.any(max_abs_attr_per_image == 0):
        print("Warning: completely 0 attributions returned for sample.")
        # If an image has 0 max abs attr, all attrs are 0 for that image
        # Divide by 1 to return the original constant 0 attributions
        max_abs_attr_per_image[torch.where(max_abs_attr_per_image == 0)] = 1.0
    # Add as many singleton dimensions to max_abs_attr_per_image as necessary to divide
    while len(max_abs_attr_per_image.shape) < len(attrs.shape):
        max_abs_attr_per_image = torch.unsqueeze(max_abs_attr_per_image, dim=-1)
    normalized = attrs / max_abs_attr_per_image
    return normalized.reshape(attrs.shape)


def _max_abs_aggregation(x):
    abs_value = x.abs()
    index = torch.argmax(abs_value, dim=1).unsqueeze(1)
    return torch.gather(x, dim=1, index=index).squeeze()


class AttributionMethod:
    def __init__(self, normalize=False, aggregation_fn=None):
        self.normalize = normalize
        aggregation_fns = {
            "avg": lambda x: torch.mean(x, dim=1),
            "max_abs": _max_abs_aggregation
        }
        self.aggregation_fn = aggregation_fns.get(aggregation_fn, None)

    def _attribute(self, x, target, **kwargs):
        raise NotImplementedError

    def __call__(self, x, target, **kwargs):
        attrs = self._attribute(x, target, **kwargs)
        if self.aggregation_fn:
            attrs = self.aggregation_fn(attrs)
        if self.normalize:
            attrs = normalize_attributions(attrs)
        return attrs


# This is not really an attribution technique, just to establish a baseline
class Random(AttributionMethod):
    def __init__(self, **kwargs):
        super(Random, self).__init__(**kwargs)

    def _attribute(self, x, target, **kwargs):
        return (torch.rand(*x.shape) * 2 - 1).to(x.device)


class EdgeDetection(AttributionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _attribute(self, x, target, **kwargs):
        device = x.device
        x = x.detach().cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        for i in range(x.shape[0]):
            for channel in range(x.shape[1]):
                x[i, channel] = sobel(x[i, channel])
        attrs = torch.tensor(x).to(device)
        return attrs


class ExpectedGradients(AttributionMethod):
    # https://github.com/suinleelab/attributionpriors

    # reference_dataset: data to load background samples from, use training data.
    def __init__(self, model, reference_dataset, n_steps=100, **kwargs):
        super().__init__(False, **kwargs)
        self.model = model
        if isinstance(reference_dataset, torch.utils.data.DataLoader):
            reference_dataset = reference_dataset.dataset
        self.reference_dataset = reference_dataset
        self.n_steps = n_steps
        self.ref_sampler = torch.utils.data.DataLoader(
            dataset=reference_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True)

    def _get_reference_batch(self, batch_size):
        self.ref_sampler = torch.utils.data.DataLoader(
            dataset=self.reference_dataset,
            batch_size=batch_size,
            shuffle=True, drop_last=True)
        return next(iter(self.ref_sampler))[0]

    def _get_sampled_input(self, inputs, references):
        batch_size = inputs.shape[0]
        k = references.shape[0]
        img_dims = inputs.shape[1:]
        alphas = torch.FloatTensor(batch_size, k).uniform_(0,1)
        alpha_shape = [batch_size,k] + [1]*len(img_dims)
        alphas = alphas.view(alpha_shape)
        inputs = inputs.unsqueeze(1)
        references = references.expand(batch_size, *references.shape)

        inputs_endpoint = inputs*alphas
        references_endpoint = references * (1.0 - alphas)
        return inputs_endpoint + references_endpoint

    def _get_grads(self, samples_input, labels, device):
        self.model.eval()
        grad_tensor = torch.zeros_like(samples_input)
        samples_input.requires_grad = True
        labels = [torch.arange(labels.shape[0]), labels]

        for i in range(self.n_steps):
            sample_slice = samples_input[:,i,...].to(device)
            output = self.model(sample_slice)
            if output.shape[1]>1:
                # if not single binary output, look at labels
                output = output[labels]
            output_grads = torch.autograd.grad(outputs=output,inputs=sample_slice,
                                               grad_outputs=torch.ones_like(output))#create_graph =True
            grad_tensor[:, i, :] = output_grads[0]
        return grad_tensor

    def _get_deltas(self, input,references):
        input_expand_mult = input.unsqueeze(1)
        sd = input_expand_mult - references
        return sd

    def _attribute(self, x, target, **kwargs):
        device = x.device
        x = x.to('cpu') # generating samples on cpu so save vram (may be not ideal for hpc)
        batch_size = x.shape[0]
        reference = self._get_reference_batch(
            self.n_steps)  # take batch_size*n_steps samples instead? does every image in batch really need different
        # background samples or is it ok to reuse the same n_steps backgrounds
        sample_input = self._get_sampled_input(x,reference)
        sample_deltas = self._get_deltas(x,reference)
        grads = self._get_grads(sample_input, target, device)
        expected_grads = sample_deltas * grads
        return expected_grads.mean(1)