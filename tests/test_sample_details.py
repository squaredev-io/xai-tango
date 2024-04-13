import sys
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from xaivision.utils import sample_details
except (Exception, ):
    raise


@pytest.fixture
def mock_model():

    class MockModel(torch.nn.Module):

        def __init__(self):
            super(MockModel, self).__init__()

        def forward(self, x):
            return torch.tensor([1.0, 2.0, 3.0])

    return MockModel()


def test_sample_details(mock_model):
    # Create a sample datapoint
    datapoint = np.ones((1, 3, 3))

    # Call the function under test
    output_data = sample_details(mock_model, datapoint)

    # Assert that output data is obtained from model and converted correctly
    assert output_data.shape == (3, )
    assert output_data.dtype == np.float64
    assert np.array_equal(output_data, np.array([1.0, 2.0, 3.0]))
