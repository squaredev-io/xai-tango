import numpy as np

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import shap

import h5py

import pandas as pd

from captum.attr import IntegratedGradients, DeepLift

from itertools import chain


class ImageDataset_normalised(torch.utils.data.Dataset):

    def __init__(self, path, nominal_laser_params=[900, 215]):
        self.path = path
        self.nominal_laser_params = np.array(nominal_laser_params).astype(
            np.float32)

    def __getitem__(self, idx):
        """Get image and target y values"""
        with h5py.File(self.path) as h5f:
            x = (np.expand_dims(np.array(h5f["x"][idx]).astype(np.float32),
                                axis=0))
            y = (np.array(h5f["y"][idx], dtype=np.float32) /
                 self.nominal_laser_params)

        image = np.array(x)

        # Get target
        target = torch.tensor(y)
        return image, target

    def __len__(self):
        with h5py.File(self.path) as h5f:
            return len(h5f['x'])


def vision_shap(data, batch_size, model_py, model_input):
    """Compute SHAP (SHapley Additive exPlanations) values for a given image.

    Args:
        - data: The dataset to be used.
        - batch_size (int): Batch size for DataLoader.
        - model_py: The PyTorch model.
        - sample_index (int): Index of the sample image for which SHAP values
        are computed.

    Returns:
        list: A list of matplotlib figures containing SHAP value plots.

    """

    ds = ImageDataset_normalised(data)

    device = torch.device('cpu')
    shap_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    background, _ = next(iter(shap_loader))
    background = background.to(device)

    model = model_py.to(device)

    explainer = shap.DeepExplainer(model, background)

    test_image = np.expand_dims(model_input, axis=0)
    test_image = torch.tensor(test_image).to(device)
    shap_values = explainer.shap_values(test_image)

    if len(np.array(shap_values).shape) == 5:
        shap_numpy = np.array(shap_values).transpose(4, 0, 2, 3, 1)
    else:
        shap_numpy = np.array(shap_values).transpose(0, 2, 3, 1)
        shap_numpy = np.expand_dims(shap_numpy, axis=0)

    test_numpy = np.array([np.array(img)
                           for img in test_image]).transpose(0, 2, 3, 1)

    plots = []
    for value in shap_numpy:
        shap.image_plot(value, test_numpy, show=False)
        plots.append(plt.gcf())
        plt.close()
    return plots


def shap_overview(data, batch_background, batch_test, model_py):
    """Compute SHAP (SHapley Additive exPlanations) overview for a given
    dataset.

    Args:
        - data: The dataset to be used.
        - batch_background (int): Batch size for background data for SHAP
        computation.
        - batch_test (int): Batch size for test data for SHAP computation.
        - model_py: The PyTorch model.

    Returns:
        list: A list of matplotlib figures containing SHAP summary plots.
    """

    ds = ImageDataset_normalised(data)

    device = torch.device('cpu')

    shap_loader = DataLoader(ds, batch_size=batch_background, shuffle=True)
    background, _ = next(iter(shap_loader))
    background = background.to(device)

    shap_tester = DataLoader(ds, batch_size=batch_test, shuffle=True)
    test_batch, _ = next(iter(shap_tester))
    test_batch = test_batch.to(device)

    model = model_py.to(device)

    explainer = shap.DeepExplainer(model, background)

    shap_values = explainer.shap_values(test_batch)

    if len(np.array(shap_values).shape) == 5:
        shap_numpy = np.array(shap_values).transpose(4, 0, 2, 3, 1)
    else:
        shap_numpy = np.array(shap_values).transpose(0, 2, 3, 1)
        shap_numpy = np.expand_dims(shap_numpy, axis=0)
    plots = []
    for i, value in enumerate(shap_numpy):
        shap_lists = []
        value_lists = []
        for j, sample in enumerate(value):
            x, y = np.squeeze(sample).shape[0], np.squeeze(sample).shape[1]
            flatten_list_shap = list(chain.from_iterable(np.squeeze(sample)))
            flatten_list_value = list(
                chain.from_iterable(np.squeeze(test_batch[j])))
            shap_lists.append(flatten_list_shap)
            value_lists.append(flatten_list_value)

        df = pd.DataFrame({
            "mean_abs_shap":
            np.mean(np.abs(np.array(shap_lists)), axis=0),
            "stdev_abs_shap":
            np.std(np.abs(np.array(shap_lists)), axis=0)
        })
        df_sorted = df.sort_values("stdev_abs_shap", ascending=False)[:10]
        shap_values = []
        pixel_values = []
        feature_names = []
        num_values = 10
        for k in range(num_values):
            pixel_num = df_sorted.index[k]
            x_unflattened, y_unflattened = int(pixel_num / x), int(pixel_num %
                                                                   y)
            name = "Pixel (" + str(x_unflattened) + "," + str(
                y_unflattened) + ")"
            shap_values.append(np.array(shap_lists)[:, pixel_num])
            pixel_values.append(np.array(value_lists)[:, pixel_num])
            feature_names.append(name)
        shap_values = np.array(shap_values).transpose(1, 0)
        pixel_values = np.array(pixel_values).transpose(1, 0)

        shap.summary_plot(shap_values,
                          pixel_values,
                          feature_names=feature_names,
                          show=False)

        plots.append(plt.gcf())
        plt.close()
    return plots


def integrated_grad(model, datasample):
    """Compute Integrated Gradients for a given data sample and model.

    Args:
        - model: The PyTorch model.
        - datasample: The input data sample.

    Returns:
        list: A list of integrated gradients for each feature.
    """

    input_py = torch.from_numpy(np.expand_dims(datasample, axis=0))
    ig = IntegratedGradients(model)
    len_targets = len(model(input_py).detach().numpy()[0])
    grads = []
    for i in range(len_targets):
        attributions = ig.attribute(input_py, target=i)
        grads.append(attributions[0][0])
    return grads


def deeplift(model, datasample):
    """Compute DeepLift attributions for a given data sample and model.

    Args:
        - model: The PyTorch model.
        - datasample: The input data sample.

    Returns:
        list: A list of DeepLift attributions for each feature.
    """

    input_py = torch.from_numpy(np.expand_dims(datasample, axis=0))
    input_py.requires_grad = True
    ig = DeepLift(model.eval())
    len_targets = len(model(input_py).detach().numpy()[0])
    dl_arrays = []
    for i in range(len_targets):
        attributions = ig.attribute(input_py, target=i).detach().numpy()
        dl_arrays.append(attributions[0][0])
    return dl_arrays
