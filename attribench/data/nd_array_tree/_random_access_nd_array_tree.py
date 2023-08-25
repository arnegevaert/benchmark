from typing import Tuple, Dict, Union, Optional, List
import os
import yaml
import numpy as np
from numpy import typing as npt
import h5py


class RandomAccessNDArrayTree:
    def __init__(
        self,
        levels: Dict[str, List[str]],
        shape: List[int],
    ):
        """
        Represents a tree of numpy NDArrays capable of writing in a
        random-access way (as opposed to append-only).
        :param levels: represents the names and keys at each level of the tree.
            E.g: {"masker": ["constant", "random"], "method": ["m1", "m2"]}
            will create a tree with 2 levels: a masker level and method level.
            Each combination of masker and method corresponds to 1 NDArray.
        """
        self.levels = levels
        self.shape = shape
        self.level_names: List[str] = sorted(list(levels.keys()))

        def _initialize_data(data, depth=0) -> Dict:
            """
            Recursively initialize the data structure.
            If we are in an internal node, a dictionary is created and keys for
            the next level are added recursively. Otherwise, a 0-valued NDArray
            of the necessary shape is created.
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

    def write_dict(
        self,
        indices: npt.NDArray,
        data: Dict,
        level_order: Optional[List[str]] = None,
    ):
        """
        Recursively write a dictionary of data to fixed indices.
        The data dictionary has to follow the same levels structure as the tree.
        For each combination of levels, data is written to the same indices.
        This is analogous to recursively looping over all keys in data and
        calling tree.write(indices, data[level1][level2]) for each NDArray.
        :param indices: Indices where data should be written for each NDArray
        :param data: Dictionary containing the data to be written
        :param level_order: List of strings indicating in which order the
            different levels are encountered in the dictionary.
            If not specified, alphabetical order is used.
        """
        if level_order is None:
            level_order = self.level_names
        elif sorted(level_order) != sorted(self.level_names):
            raise ValueError(f"Invalid level_order: {level_order}")

        def _write_rec(_data, level_keys):
            depth = len(level_keys.keys())
            if depth < len(self.level_names):
                cur_level_name = level_order[depth]
                for key in _data.keys():
                    # _data is a dictionary, loop over all keys and recurse
                    level_keys[cur_level_name] = key
                    _write_rec(_data[key], dict(level_keys))
            else:
                # We have reached a leaf, _data is an NDArray.
                # Write to the specified location.
                self.write(indices, _data, **level_keys)

        _write_rec(data, {})

    def write_dict_split(
        self,
        indices_dict: Dict[str, npt.NDArray],
        target_indices: npt.NDArray,
        split_level: str,
        data: Dict,
        level_order: Optional[List[str]] = None,
    ):
        """
        Same as write_dict, but instead of writing the full NDArray to each
        location, indices in each NDArray are chosen based on some level.
        For example: write indices [1,2,4] to all locations with
        method_name == "Gradient", and indices [3, 5] to all locations with
        method_name == "InputXGradient".
        :param indices_dict: Dictionary mapping each level key of the split
            level to its corresponding indices in the original data.
            Do not confuse with indices in write or write_dict: these are
            target indices (indices where the data should be written to).
        :param target_indices: NDArray containing the indices where the data
            should be written to.
        :param split_level: The level at which the NDArrays must be split.
            This level is not present in data.
        :param data: Dictionary containing the data to be written
        :param level_order: List of strings indicating in which order the
            different levels are encountered in the dictionary.
            If not specified, alphabetical order is used.
        """
        # This contains the levels that should be present in the data dict
        # and in level_order. These are all levels except the split level.
        data_levels = list(self.level_names)
        data_levels.remove(split_level)

        if level_order is None:
            level_order = data_levels
        elif sorted(level_order) != sorted(data_levels):
            raise ValueError(f"Invalid level_order: {level_order}")

        def _write_rec(_data, level_keys, split_key):
            depth = len(level_keys.keys())
            if depth < len(self.level_names):
                # We index using depth-1 because level_order does not contain
                # the split level.
                cur_level_name = level_order[depth - 1]
                for key in _data.keys():
                    # Loop over all keys and recurse
                    level_keys[cur_level_name] = key
                    _write_rec(_data[key], dict(level_keys), split_key)
            else:
                # We have reached a leaf, _data is an NDArray.
                # Write to the specified location.
                cur_indices = indices_dict[split_key]
                cur_target_indices = target_indices[cur_indices]
                self.write(
                    cur_target_indices, _data[cur_indices], **level_keys
                )

        # Write separately for each split key
        for key in indices_dict.keys():
            # We basically start at depth 1, the split_level is already set.
            # This level will not be present in the data dictionary.
            _write_rec(data, {split_level: key}, split_key=key)

    def write(
        self, indices: npt.NDArray, data: npt.NDArray, **level_keys
    ) -> None:
        """
        Writes data to the given indices (axis=0) of a certain NDArray.
        The NDArray is given by the level_keys: for each level name there
        should be 1 key. This encodes the path to take down the tree.
        :param indices: Indices of NDArray where data should be written
        :param data: The data to write in the NDArray at the given indices
        :param level_keys: a key for each level to indicate the path to the
            desired NDArray
        """
        if set(level_keys.keys()) != set(self.levels.keys()):
            raise ValueError(
                f"Must provide key for all levels: {list(self.levels.keys())}"
                "(received {level_keys})"
            )
        dest: Union[Dict, npt.NDArray] = self._data
        # Find the relevant NDArray
        for level_name in self.level_names:
            dest = dest[level_keys[level_name]]
        # At this point, dest should be an NDArray. Write the data.
        dest[indices] = data

    def get(self, **level_keys) -> npt.NDArray:
        """
        Get a single NDArray from the tree
        :param level_keys: Keys encoding the path to take down the tree
        """
        if set(level_keys.keys()) != set(self.levels.keys()):
            raise ValueError(
                f"Must provide key for all levels: {list(self.levels.keys())}"
            )
        res = self._data
        for level_name in self.level_names:
            res = res[level_keys[level_name]]
        if not isinstance(res, np.ndarray):
            raise ValueError(
                f"Expected leaf node to be ndarray, got {type(res)}"
            )
        return res

    def save_to_hdf(self, group: h5py.Group) -> None:
        """
        Attach the NDArrayTree to an HDF5 group
        :param group: the HDF5 group to which the root of this NDArrayTree
            should be attached
        """

        def _add_rec(cur_data, cur_group, depth=0):
            for key in cur_data:
                cur_group.attrs["level_name"] = self.level_names[depth]
                if depth < len(self.level_names) - 1:
                    next_group = cur_group.create_group(key)
                    _add_rec(cur_data[key], next_group, depth + 1)
                else:
                    cur_group.create_dataset(key, data=cur_data[key])

        _add_rec(self._data, group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> "RandomAccessNDArrayTree":
        def _load_metadata_rec(
            node: Union[h5py.Group, h5py.Dataset], cur_levels
        ) -> Tuple[Dict, List[int]]:
            """
            Recursively loads the necessary metadata from the tree:
            level names, level keys, NDArray shape.
            :return: the necessary arguments to instantiate an NDArrayTree from
                this file.
            """
            if isinstance(node, h5py.Group):
                keys = list(node.keys())
                level_name = node.attrs["level_name"]
                cur_levels[level_name] = keys
                next_node = node[keys[0]]
                assert isinstance(next_node, (h5py.Group, h5py.Dataset))
                return _load_metadata_rec(next_node, cur_levels)
            elif isinstance(node, h5py.Dataset):
                return cur_levels, list(node.shape)

        def _copy_ndarrays_rec(h5py_node: h5py.Group, tree_node: Dict):
            """
            Recursively copy the NDArrays from the HDF5 file to the tree
            """
            for key in list(h5py_node.keys()):
                next_node = h5py_node[key]
                if isinstance(next_node, h5py.Dataset):
                    # We are at the bottom, copy the datasets to the tree
                    # An HDF5 Dataset can be converted to an NDArray using [()]
                    tree_node[key] = next_node[()]
                elif isinstance(next_node, h5py.Group):
                    # We are at an inner node, descend
                    _copy_ndarrays_rec(next_node, tree_node[key])
                else:
                    raise ValueError(f"Invalid type: {type(next_node)}")

        # Instantiate the NDArrayTree with the necessary metadata
        levels, shape = _load_metadata_rec(group, {})
        result = cls(levels, shape)

        # Traverse the NDArrayTree to copy the NDArrays to their respective
        # locations
        _copy_ndarrays_rec(group, result._data)
        return result

    def save_to_dir(self, path: str) -> None:
        """
        Save the NDArrayTree to a directory of CSV files with metadata in
        a yaml file.
        :param path: The path to the directory
        """
        # First read any existing metadata
        with open(f"{path}/metadata.yaml", "r") as fp:
            metadata = yaml.safe_load(fp) or {}

        # Add the necessary metadata
        metadata["level_names"] = self.level_names
        metadata["levels"] = self.levels
        metadata["shape"] = self.shape

        # Write the metadata
        with open(f"{path}/metadata.yaml", "w") as fp:
            yaml.safe_dump(metadata, fp)

        # Write the NDArrays
        def _add_rec(cur_data, cur_path, depth=0):
            for key in cur_data:
                if depth < len(self.level_names) - 1:
                    next_path = os.path.join(cur_path, key)
                    _add_rec(cur_data[key], next_path, depth + 1)
                else:
                    if not os.path.exists(cur_path):
                        os.makedirs(cur_path, exist_ok=True)
                    np.savetxt(
                        os.path.join(cur_path, f"{key}.csv"), cur_data[key]
                    )

        _add_rec(self._data, path)

    def merge(
        self,
        other: "RandomAccessNDArrayTree",
        level: str,
        allow_overwrite: bool,
    ) -> None:
        def _merge_rec(cur_data, other_data, depth=0):
            level_name = self.level_names[depth]
            if level_name == level:
                intersection = set(cur_data.keys()) & set(other_data.keys())
                if not allow_overwrite and len(intersection) > 0:
                    raise ValueError(
                        f"Cannot merge: {level_name} has overlapping entries: "
                        f"{intersection}. Set allow_overwrite=True to allow "
                        f"overwriting."
                    )
                for key in other_data:
                    cur_data[key] = other_data[key]
            else:
                for key in cur_data:
                    _merge_rec(cur_data[key], other_data[key], depth + 1)

        _merge_rec(self._data, other._data)
        self.levels[level] += other.levels[level]
        # Remove duplicates that might be introduced from overlapping keys
        self.levels[level] = sorted(list(set(self.levels[level])))

    @classmethod
    def load_from_dir(cls, path: str) -> "RandomAccessNDArrayTree":
        # Load metadata
        with open(os.path.join(path, "metadata.yaml"), "r") as fp:
            metadata = yaml.safe_load(fp)

        # Instantiate the NDArrayTree with the necessary metadata
        result = cls(metadata["levels"], metadata["shape"])

        # Traverse the NDArrayTree to copy the NDArrays to their respective
        # locations
        def _copy_ndarrays_rec(cur_path, tree_node):
            """
            Recursively copy the NDArrays from the directory to the tree
            """
            for key in os.listdir(cur_path):
                next_path = os.path.join(cur_path, key)
                if os.path.isdir(next_path):
                    # We are at an inner node, descend
                    _copy_ndarrays_rec(next_path, tree_node[key])
                elif key.endswith(".csv"):
                    # We are at the bottom, copy the datasets to the tree
                    tree_node[key[:-4]] = np.loadtxt(next_path)

        _copy_ndarrays_rec(path, result._data)
        return result
