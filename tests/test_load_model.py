import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from xaivision.utils import load_models


@pytest.fixture
def onnx_model(tmp_path):
    # Create a temporary ONNX model file for testing
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(),
                                torch.nn.Linear(5, 1))
    dummy_input = torch.randn(1, 10)
    torch.onnx.export(model, dummy_input, str(tmp_path / 'test_model.onnx'))
    return str(tmp_path / 'test_model.onnx')


def test_load_models(onnx_model):
    # Load the ONNX model using the load_model function
    pytorch_model = load_models(onnx_model)

    # Verify that the loaded model is an instance of torch.nn.Module
    assert isinstance(pytorch_model, torch.nn.Module)

    # Verify that the loaded model has the expected architecture
    layers = list(pytorch_model.children())
    assert len(layers) == 3

    assert isinstance(layers[0], torch.nn.Linear)
    assert isinstance(layers[1], torch.nn.ReLU)
    assert isinstance(layers[2], torch.nn.Linear)
