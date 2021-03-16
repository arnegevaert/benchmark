import torch


class ExpectedGradients:
    # https://github.com/suinleelab/attributionpriors

    # reference_dataset: data to load background samples from, use training data.
    def __init__(self, model, reference_dataset, num_samples):
        self.model = model
        if isinstance(reference_dataset, torch.utils.data.DataLoader):
            reference_dataset = reference_dataset.dataset
        self.reference_dataset = reference_dataset
        self.num_samples = num_samples
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
        alphas = torch.FloatTensor(batch_size, k).uniform_(0, 1)
        alpha_shape = [batch_size, k] + [1] * len(img_dims)
        alphas = alphas.view(alpha_shape)
        inputs = inputs.unsqueeze(1)
        references = references.expand(batch_size, *references.shape)

        inputs_endpoint = inputs * alphas
        references_endpoint = references * (1.0 - alphas)
        return inputs_endpoint + references_endpoint

    def _get_grads(self, samples_input, labels, device):
        self.model.eval()
        grad_tensor = torch.zeros_like(samples_input)
        if not samples_input.requires_grad:
            samples_input.requires_grad = True
        labels = [torch.arange(labels.shape[0]), labels]

        for i in range(self.num_samples):
            sample_slice = samples_input[:, i, ...].to(device)
            output = self.model(sample_slice)
            if output.shape[1] > 1:
                # if not single binary output, look at labels
                output = output[labels]
            output_grads = torch.autograd.grad(outputs=output, inputs=sample_slice,
                                               grad_outputs=torch.ones_like(output))  # create_graph =True
            grad_tensor[:, i, :] = output_grads[0]
        return grad_tensor

    def _get_deltas(self, input, references):
        input_expand_mult = input.unsqueeze(1)
        sd = input_expand_mult - references
        return sd

    def __call__(self, x, target):
        device = x.device
        x = x.to('cpu')  # generating samples on cpu so save vram (may be not ideal for hpc)
        reference = self._get_reference_batch(
            self.num_samples)  # take batch_size*n_steps samples instead? does every image in batch really need different
        # background samples or is it ok to reuse the same n_steps backgrounds
        sample_input = self._get_sampled_input(x, reference)
        sample_deltas = self._get_deltas(x, reference)
        grads = self._get_grads(sample_input, target, device)
        expected_grads = sample_deltas * grads
        return expected_grads.mean(1).detach().cpu()
