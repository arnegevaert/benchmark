import h5py


def _check_is_dataset(ds) -> h5py.Dataset:
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"Expected ds to be a Dataset, but got {type(ds)}")
    return ds
