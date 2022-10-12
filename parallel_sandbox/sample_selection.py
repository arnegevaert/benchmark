from util.get_dataset_model import get_dataset, get_model
from attrbench.distributed import SampleSelection

if __name__ == "__main__":
    sample_selection = SampleSelection(get_model, get_dataset(), "test.h5", num_samples=128,
                                       sample_shape=(3, 224, 224), batch_size=16)
    sample_selection.start()