from util.get_dataset_model import get_dataset, get_model
from attrbench.distributed import AttributionsComputation
from attrbench.data import HDF5Dataset, AttributionsDatasetWriter
from captum import attr


def method_factory(model):
    saliency = attr.Saliency(model)
    ixg = attr.InputXGradient(model)
    ig = attr.IntegratedGradients(model)
    return {
        "Gradient": saliency.attribute,
        "InputXGradient": ixg.attribute,
        "IntegratedGradients": lambda x, y: ig.attribute(inputs=x, target=y, internal_batch_size=1)
    }


if __name__ == "__main__":
    dataset = HDF5Dataset("samples.h5")
    writer = AttributionsDatasetWriter("attributions.h5", truncate=True, num_samples=len(dataset),
                                       sample_shape=dataset.sample_shape)
    computation = AttributionsComputation(get_model, method_factory, dataset, batch_size=16, writer=writer)
    computation.start()
