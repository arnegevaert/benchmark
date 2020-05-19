from typing import Iterable, Callable, List, Dict
from collections import OrderedDict
import itertools
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr

metrics = {'ssim': ssim,
           'spearman': spearmanr}


def model_randomisation(data: Iterable, model: Callable, methods: Dict[str, Callable],
                        n_batches=None, layer_list=None, device: str = "cpu", metric='spearman'):
    # layer_list: list of names and modules to randomise, like model.named_children
    if not layer_list:
        layer_list = list(model.named_children())
    metric_function = metrics[metric]
    # get normal model attributions
    baseline_attributions = _attribution_loop(data, methods,n_batches, device)

    # get attributions with layers weights randomised starting from output to input
    layer_name_list = []
    rand_attrs = {}  # dict where key = name of layer randomised, value = dict of attributions per method
    for name, module in reversed(layer_list):
        if not _randomise_module(module): #returns true on succes
            # no need to do attributions, keeps data cleaner
            continue
        layer_name_list.append(name) # keep track of names of layers that actually have been reset
        rand_attrs[name] = _attribution_loop(data, methods,n_batches, device)

    # collect metric values from attributes for each method and each layer depth
    metrics_result = {}
    for m_name in methods:
        metric_per_layer = OrderedDict()
        for l_name in layer_name_list:
            metric_per_layer[l_name] = _get_metric_results(baseline_attributions[m_name],
                                                         rand_attrs[l_name][m_name], metric_fc=metric_function)
        metrics_result[m_name] = metric_per_layer

    return metrics_result


def _randomise_module(module):
    has_learnable_parameters = False
    for _, child_module in module.named_modules():
        if hasattr(child_module, 'reset_parameters'):
            child_module.reset_parameters()
            has_learnable_parameters = True
    return has_learnable_parameters


def _get_metric_results(baseline: list, rand: list, metric_fc):
    results = []
    for i in range(len(baseline)):
        results.append(metric_fc(baseline[i], rand[i]))
    return np.mean(results)


def _attribution_loop(data, methods, n_batches, device):

    attributions_dict = {m_name: [] for m_name in methods}
    data_slice = itertools.islice(data, n_batches)
    for b, (samples, labels) in enumerate(data_slice):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for m_name in methods:
            method = methods[m_name]
            attr = method(samples, target=labels)
            flattened_attrs = attr.reshape(attr.shape[0], -1)
            attributions_dict[m_name].extend(flattened_attrs.cpu().detach().numpy())
    return attributions_dict
