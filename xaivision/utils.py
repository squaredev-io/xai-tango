import torch
from onnx2torch import convert
import onnxruntime as ort
import numpy as np
from torchinfo import summary
from torchviz import make_dot

import h5py
import torch.nn as nn

import onnx

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from nmf_func import NMF
except (Exception, ):
    raise


class MedPCacheDataset_normalised():
    """
    Dataset interface of RAISE-LPBF-Laser benchmark cache single frame power
    prediction.
    """

    def __init__(self, cache_fp, nominal_laser_params=[900, 215], **_):
        self.cache_fp = cache_fp
        self.nominal_laser_params = np.array(nominal_laser_params).astype(
            np.float32)

        with h5py.File(self.cache_fp, "r") as h5f:
            self._len = len(h5f["x"])

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        with h5py.File(self.cache_fp, "r") as h5f:
            x = (np.expand_dims(np.array(h5f["x"][index]).astype(np.float32),
                                axis=0))
            y = (np.array(h5f["y"][index], dtype=np.float32) /
                 self.nominal_laser_params)
        return x, y


class YourModelWithoutLastLayers(nn.Module):

    def __init__(self, original_model, remove_layers):
        super(YourModelWithoutLastLayers, self).__init__()

        self.features = nn.Sequential(
            *list(original_model.children())[:-1 * remove_layers])

    def forward(self, x):
        # Forward pass through the modified layers
        x = self.features(x)
        return x


def load_models(model):
    """
    Load model and convert it to Python format.

    Parameters:
        - model (str): Path to the model or the actual model loaded

    Returns:
        Model in Python format.

    .. todo::
        -
    """
    model_py = convert(model)
    return model_py


def load_sample(data, sample):
    """
    Load a sample from the dataset.

    Parameters:
        - data (str): Path to the dataset or the actual dataset
        - sample (int): Index of the sample to load.
    Returns:
        object: The loaded datapoint.
    """

    ds = MedPCacheDataset_normalised(data)
    datapoint = ds.__getitem__(sample)

    return datapoint


def check_onnx_torch_out(onnx_model_path, torch_model, model_input):
    """
    Compare outputs of an ONNX model and a PyTorch model given a data point.

    Parameters:
        - onnx_model_path (str): Path to the ONNX model file.
        - torch_model (torch.nn.Module): PyTorch model.
        - datapoint (tuple): A tuple containing input data and its
        corresponding labels.

    Returns:
        None
    """

    data_inp = np.expand_dims(model_input, axis=0)

    # for layers in dict(torch_model.named_modules()): print (layers)

    data_torch = torch.from_numpy(data_inp)
    torch_model.eval()
    out_torch = torch_model(data_torch)

    # output =[node.name for node in model.graph.output]
    onnx_model = onnx.load(onnx_model_path)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))

    # print('Inputs: ', net_feed_input[0])
    # print('Outputs: ', output[0])
    ort_sess = ort.InferenceSession(onnx_model_path)
    outputs_ort = ort_sess.run(None, {net_feed_input[0]: data_inp})

    diagnosis = "Torch output : " + str(out_torch) + "\n"
    diagnosis = diagnosis + "ONNX output : " + str(outputs_ort) + "\n"
    diagnosis = diagnosis + "-------------------------------\n"
    diagnosis = diagnosis + "Difference between onnx and pytorch model: "
    diagnosis = diagnosis + str(
        torch.max(torch.abs(torch.from_numpy(outputs_ort[0]) - out_torch)))
    # print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-7))
    return diagnosis


def check_model_data_compatibility(model, data_size, output_size):
    # create some sample input data
    x = np.expand_dims(torch.randn(data_size), axis=0)
    x = torch.from_numpy(x)
    # generate predictions for the sample data
    y = model(x).squeeze(0).detach().numpy()
    return y.shape == output_size


def model_details(model, data_size):
    """
    Retrieve details about the model including architecture visualization and
    summary.

    Parameters:
        - model (torch.nn.Module): The PyTorch model to inspect.
        - data_size (tuple): The size of the input data (e.g., (channels,
        height, width)).

    Returns:
        tuple: A tuple containing two elements:
            - dot (graphviz.Digraph): Graph visualization of the model
            architecture.
            - sum (str): Summary of the model including the number of
            parameters and layers.
    """

    # create some sample input data
    x = np.expand_dims(torch.randn(data_size), axis=0)
    x = torch.from_numpy(x)
    # generate predictions for the sample data
    y = model(x)

    # generate a model architecture visualization
    dot = make_dot(y.mean(),
                   params=dict(model.named_parameters()),
                   show_attrs=True,
                   show_saved=True)

    sum = summary(model, input_size=x.shape, verbose=0)

    model_output = y.squeeze(0).detach().numpy()
    model_input_txt = "Model input shape: " + str(data_size)
    model_output_txt = "Model output shape: " + str(model_output.shape)
    summary_with_text = str(
        sum) + "\n" + model_input_txt + "\n" + model_output_txt
    summary_with_text = summary_with_text + "\n" + "=\
=======================================================================\
=================="

    return dot, summary_with_text


def sample_details(model, datapoint):
    """
    Get details about a sample by passing it through the model.

    Parameters:
        - model (torch.nn.Module): The PyTorch model to use.
        - datapoint (numpy.ndarray): The input data point.

    Returns:
        tuple: A tuple containing two elements:
            - input_data (numpy.ndarray): The input data after squeezing.
            - output_data (numpy.ndarray): The output data from the model
            after squeezing, converting to numpy array,
              and detaching from the computation graph.
    """

    data_inp = np.expand_dims(datapoint, axis=0)
    data_torch = torch.from_numpy(data_inp)
    model.eval()
    out_torch = model(data_torch).squeeze(0).detach().numpy().astype(
        np.float64)

    return out_torch


def conv2d_feature_vis_no_extra_layers(model, datasample):
    """
    Extracts and visualizes feature maps from Conv2d layers in a given model
    without additional layers.

    Parameters:
        - model (torch.nn.Module): The PyTorch model from which to extract
        feature maps.
        - datasample (tuple): A tuple containing input data and its
        corresponding labels.

    Returns:
        tuple: A tuple containing two elements:
            - processed (list): A list of processed feature maps.
            - names (list): A list of names of the Conv2d layers whose feature
            maps were extracted.
    """

    model_weights = []
    conv_layers = []

    model_children = list(model.children())
    counter = 0

    for child in range(len(model_children)):
        if type(model_children[child]) is nn.Conv2d:
            counter += 1
            model_weights.append(model_children[child].weight)
            conv_layers.append(model_children[child])
        elif type(model_children[child]) is nn.Sequential:
            for i in range(len(model_children[child])):
                for c in model_children[child][i].children():
                    if type(c) is nn.Conv2d:
                        counter += 1
                        model_weights.append(c.weight)
                        conv_layers.append(c)

    outputs = []
    names = []
    data_inp = np.expand_dims(datasample, axis=0)
    image = torch.from_numpy(data_inp)

    for layer in conv_layers:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    # print feature_maps
    # for feature_map in outputs:
    #     print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    # for fm in processed:
    #     print(fm.shape)

    return processed, names


def find_convolutions(model):
    """
    Finds convolutional layers within a PyTorch model.

    Args:
        - model (torch.nn.Module): The PyTorch model to search for
        convolutional layers.

    Returns:
        tuple: A tuple containing two lists:
            - The indices of convolutional layers found in the model.
            - The layers containing convolutional layers found in the model.
    """

    model_children = list(model.children())
    layers = []
    spot_convs = []

    for layer_cnt, child in enumerate(range(len(model_children))):
        if type(model_children[child]) is nn.Sequential:
            for i in range(len(model_children[child])):
                for cnt, c in enumerate(model_children[child][i].children()):
                    layers.append(model_children[child])
                    if type(c) is nn.Conv2d:
                        spot_convs.append(cnt)
        else:
            layers.append(model_children[child])
            if type(model_children[child]) is nn.Conv2d:
                spot_convs.append(layer_cnt)

    return spot_convs, layers


def conv2d_feature_vis_extra_layers(model, datasample):
    """
    Extracts and visualizes feature maps from Conv2d layers in a given model
    with additional layers.

    Parameters:
        - model (torch.nn.Module): The PyTorch model from which to extract
        feature maps.
        - datasample (tuple): A tuple containing input data and its
        corresponding labels.

    Returns:
        tuple: A tuple containing two elements:
            - processed (list): A list of processed feature maps.
            - names (list): A list of names of the Conv2d layers whose feature
            maps were extracted.
    """

    spot_convs, layers = find_convolutions(model)

    conv_outputs = []
    conv_names = []
    data_inp = np.expand_dims(datasample, axis=0)
    image = torch.from_numpy(data_inp)

    for i, layer in enumerate(layers[:spot_convs[-1] + 1]):
        image = layer(image)
        if i in spot_convs:
            conv_outputs.append(image)
            conv_names.append(str(layer))

    processed = []
    for feature_map in conv_outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    return processed, conv_names


def find_components(model, datasample, components):
    """
    Find components using Non-Negative Matrix Factorization (NMF) based on
    model activations.

    Args:
        - model (torch.nn.Module): The neural network model.
        - datasample (tuple): A tuple containing input data and its
        corresponding label, Expected format: (input_data, label).
        - components (int): The number of components to extract.

    Returns:
        numpy.ndarray: Heatmaps of the components for the given input data.

    Raises:
        ValueError: If the components argument is not a positive integer.
    """
    data_inp = np.expand_dims(datasample, axis=0)
    image = torch.from_numpy(data_inp)

    spot_convs, layers = find_convolutions(model)
    remove_until_conv = len(layers) - (spot_convs[-1] + 1)

    new_model = YourModelWithoutLastLayers(model, remove_until_conv)
    features = new_model(image)

    flat_features = features.permute(0, 2, 3, 1).contiguous().view(
        (-1, features.size(1)))  # NxCxHxW -> (N*H*W)xC

    K = components
    with torch.no_grad():
        W, _ = NMF(flat_features, K, random_seed=0, cuda=False, max_iter=50)

    heatmaps = W.cpu().view(features.size(0), features.size(2),
                            features.size(3),
                            K).permute(0, 3, 1, 2)  # (N*H*W)xK -> NxKxHxW
    heatmaps = torch.nn.functional.interpolate(
        heatmaps,
        size=np.squeeze(datasample).shape,
        mode='bilinear',
        align_corners=False)
    heatmaps /= heatmaps.max(dim=3, keepdim=True)[0].max(
        dim=2, keepdim=True)[0]  # normalize by factor (i.e., 1 of K)
    heatmaps = heatmaps.cpu().numpy()

    return heatmaps
