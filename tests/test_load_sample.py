import pytest
import numpy as np

import h5py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from xaivision.utils import MedPCacheDataset_normalised, load_sample


# Define test cases for MedPCacheDataset_normalised
class TestMedPCacheDatasetNormalised:

    @pytest.fixture
    def dataset(self, tmp_path):
        data_path = tmp_path / "test_data.h5"
        with h5py.File(data_path, "w") as h5f:
            h5f.create_dataset("x", data=np.random.rand(10, 128, 128))
            h5f.create_dataset("y", data=np.random.rand(10))
        return data_path

    def test_len(self, dataset):
        dataset_obj = MedPCacheDataset_normalised(dataset)
        assert len(dataset_obj) == 10

    def test_getitem(self, dataset):
        dataset_obj = MedPCacheDataset_normalised(dataset)
        x, y = dataset_obj[0]
        assert x.shape == (1, 128, 128)
        assert isinstance(y, np.ndarray)
        assert y.shape == (2, )  # Assuming nominal_laser_params is length 2


# Define test cases for load_sample
class TestLoadSample:

    @pytest.fixture
    def dataset(self, tmp_path):
        data_path = tmp_path / "test_data.h5"
        with h5py.File(data_path, "w") as h5f:
            h5f.create_dataset("x", data=np.random.rand(10, 128, 128))
            h5f.create_dataset("y", data=np.random.rand(10, 2))
        return data_path

    def test_load_sample(self, dataset):

        sample_index = 0
        datapoint = load_sample(dataset, sample_index)
        x, y = datapoint
        assert x.shape == (1, 128, 128)
        assert y.shape == (2, )
        assert isinstance(y[0], np.float32)
        assert isinstance(y[1], np.float32)
