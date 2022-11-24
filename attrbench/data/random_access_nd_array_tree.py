from typing import List, Tuple, Dict, Union
import numpy as np
from numpy import typing as npt
import h5py


class RandomAccessNDArrayTree:
    def __init__(self, levels: Dict[str, Tuple[str]], shape: Tuple[int, ...]):
        """
        Represents a tree of numpy NDArrays capable of writing in a random-access way
        (as opposed to append-only).
        :param levels: represents the names and keys at each level of the tree.
            E.g: {"masker": ["constant", "random"], "method": ["m1", "m2"]}
            will create a tree with 2 levels: a masker level and method level.
            Each combination of masker and method corresponds to 1 NDArray.
        """
        self.levels = levels
        self.shape = shape
        self.level_names = sorted(list(levels.keys()))

        def _initialize_data(data, depth=0) -> Dict:
            """
            Recursively initialize the data structure.
            If we are in an internal node, a dictionary is created and keys for
            the next level are added recursively. Otherwise, a 0-valued NDArray of the necessary
            shape is created.
            :param data: The nested dictionary with the data we are constructing
            :param depth: The current depth
            :return: Nested dictionary containing the tree of NDArrays
            """
            level_keys = levels[self.level_names[depth]]
            for key in level_keys:
                if depth < len(levels) - 1:
                    data[key] = {}
                    _initialize_data(data[key], depth + 1)
                else:
                    data[key] = np.zeros(shape=self.shape)
            return data
        self._data: Dict = _initialize_data({})

    def write(self, indices: npt.NDArray, data: npt.NDArray, **level_keys) -> None:
        """
        Writes data to the given indices (axis=0) of a certain NDArray.
        The NDArray is given by the level_keys: for each level name there should be 1 key.
        This encodes the path to take down the tree.
        :param indices: Indices of NDArray where data should be written
        :param data: The data to write in the NDArray at the given indices
        :param level_keys: a key for each level to indicate the path to the desired NDArray
        """
        if set(level_keys.keys()) != set(self.levels.keys()):
            raise ValueError(f"Must provide key for all levels: {list(self.levels.keys())}")
        dest: Union[Dict, npt.NDArray] = self._data
        # Find the relevant NDArray
        for level_name in self.level_names:
            dest = dest[level_keys[level_name]]
        # At this point, dest should be an NDArray. Write the data.
        dest[indices] = data

    def add_to_hdf(self, group: h5py.Group) -> None:
        """
        Attach the NDArrayTree to an HDF5 group
        :param group: the HDF5 group to which the root of this NDArrayTree should be attached
        """
        def _add_rec(cur_data, cur_group, depth=0):
            for key in cur_data:
                cur_group.attrs["level_name"] = self.level_names[depth]
                if depth < len(self.level_names) - 1:
                    next_group = cur_group.create_group(key)
                    _add_rec(cur_data[key], next_group, depth+1)
                else:
                    cur_group.create_dataset(key, data=cur_data[key])
        _add_rec(self._data, group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> "RandomAccessNDArrayTree":
        def _load_metadata(node: Union[h5py.Group, h5py.Dataset], cur_levels) -> Tuple[Dict, Tuple[int, ...]]:
            """
            Loads the necessary metadata from the tree: level names, level keys, NDArray shape.
            :return: the necessary arguments to instantiate an NDArrayTree from this file.
            """
            if isinstance(node, h5py.Group):
                keys = list(node.keys())
                level_name = node.attrs["level_name"]
                cur_levels[level_name] = keys
                return _load_metadata(node[keys[0]], cur_levels)
            elif isinstance(node, h5py.Dataset):
                return cur_levels, node.shape

        def _copy_ndarrays(h5py_node: h5py.Group, tree_node: Dict):
            for key in list(h5py_node.keys()):
                if isinstance(h5py_node[key], h5py.Dataset):
                    # We are at the bottom, copy the datasets to the tree
                    # An HDF5 Dataset can be converted to an NDArray using [()]
                    tree_node[key] = h5py_node[key][()]
                else:
                    # We are at an inner node, descend
                    _copy_ndarrays(h5py_node[key], tree_node[key])

        # Instantiate the NDArrayTree with the necessary metadata
        levels, shape = _load_metadata(group, {})
        result = cls(levels, shape)

        # Traverse the NDArrayTree to copy the NDArrays to their respective locations
        _copy_ndarrays(group, result._data)
        return result

    def get(self, **level_keys) -> npt.NDArray:
        """
        Get a single NDArray from the tree
        :param level_keys: Keys encoding the path to take down the tree
        """
        if set(level_keys.keys()) != set(self.levels.keys()):
            raise ValueError(f"Must provide key for all levels: {list(self.levels.keys())}")
        res = self._data
        for level_name in self.level_names:
            res = res[level_keys[level_name]]
        return res
