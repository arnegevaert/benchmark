from util.get_dataset_model import get_dataset, get_model
from attrbench.data import HDF5DatasetWriter
from attrbench.distributed import SampleSelection

if __name__ == "__main__":
    num_samples = 128
    writer = HDF5DatasetWriter(path="samples.h5", num_samples=num_samples, sample_shape=(3, 224, 224))
    sample_selection = SampleSelection(get_model, get_dataset(), writer, num_samples=num_samples, batch_size=16)
    sample_selection.start()