from os import path
import pandas as pd
import numpy as np
import h5py


# TODO this class should be used by Suite to keep track of its results
class Result:
    def __init__(self, data, metadata, num_samples, images=None, attributions=None):
        self.data = data
        self.metadata = metadata
        self.num_samples = num_samples
        self.images = images
        self.attributions = attributions

    @staticmethod
    def load_hdf(filename):
        if path.isfile(filename):
            images, attributions = None, None
            with h5py.File(filename, "r") as fp:
                if "images" in fp.keys():
                    images = np.array(fp["images"])
                if "attributions" in fp.keys():
                    attributions = {
                        method: np.array(fp["attributions"][method])
                        for method in fp["attributions"].keys()
                    }
                data = {
                    metric: {
                        method: pd.DataFrame(fp["results"][metric][method])
                        for method in fp["results"][metric].keys()
                    } for metric in fp["results"].keys()
                }
                metadata = {
                    metric: {
                        key: fp["results"][metric].attrs[key]
                        for key in fp["results"][metric].attrs.keys()
                    } for metric in fp["results"].keys()
                }
                num_samples = fp["results"].attrs["num_samples"]
            return Result(data, metadata, num_samples, images, attributions)
        else:
            raise ValueError(f"File {filename} does not exist")

    def get_metrics(self):
        return list(self.data.keys())

    def get_methods(self):
        return list(self.data[self.get_metrics()[0]].keys())
