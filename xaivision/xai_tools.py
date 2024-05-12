import numpy as np

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import shap

import h5py

import pandas as pd

from captum.attr import IntegratedGradients, DeepLift

from itertools import chain

from tqdm import tqdm

import random

import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from xai_vision.xaivision.utils import (
        full_squeeze)
except (Exception, ):
    raise


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
    return plots, shap_numpy


def connected_components(image):
    """
    Finds the largest connected component in a binary image.

    Args:
    - image (list): A binary image represented as a list of lists,
      where each inner list represents a row of pixels and each pixel
      has a value of either 0 or 1.

    Returns:
    - set: A set of pixel coordinates belonging to the largest connected
        component.
    """

    height = len(image)
    width = len(image[0])

    # Function to get neighbors of a pixel
    def get_neighbors(x, y):
        """
        Gets the neighboring pixel coordinates of a given pixel.

        Args:
        - x (int): X-coordinate of the pixel.
        - y (int): Y-coordinate of the pixel.

        Returns:
        - list: List of neighboring pixel coordinates.
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < height and 0 <= ny < width:
                    neighbors.append((nx, ny))
        return neighbors

    # Function to perform DFS traversal to find connected components
    def dfs(x, y, label):
        """
        Performs depth-first search traversal to find connected components.

        Args:
        - x (int): X-coordinate of the starting pixel.
        - y (int): Y-coordinate of the starting pixel.
        - label (int): Label of the connected component.

        Returns:
        - int: Size of the connected component.
        """
        stack = [(x, y)]
        size = 0
        while stack:
            cx, cy = stack.pop()
            if visited[cx][cy]:
                continue
            visited[cx][cy] = True
            labels[cx][cy] = label
            size += 1
            for nx, ny in get_neighbors(cx, cy):
                if image[nx][ny] == 1 and not visited[nx][ny]:
                    stack.append((nx, ny))
        return size

    # Initialize data structures
    visited = [[False] * width for _ in range(height)]
    labels = [[0] * width for _ in range(height)]
    label_count = 0
    component_sizes = {}

    # Perform connected component labeling
    for i in range(height):
        for j in range(width):
            if image[i][j] == 1 and not visited[i][j]:
                label_count += 1
                size = dfs(i, j, label_count)
                component_sizes[label_count] = size

    # Find the largest component
    largest_component_label = max(component_sizes, key=component_sizes.get)

    # Create a set of pixels belonging to the largest component
    largest_component_pixels = set()
    for i in range(height):
        for j in range(width):
            if labels[i][j] == largest_component_label:
                largest_component_pixels.add((i, j))

    return largest_component_pixels


def zero_non_largest_components(image, largest_component_pixels):
    """
    Zeros out pixels in the image that do not belong to the largest connected
    component.

    Args:
    - image (list): A binary image represented as a list of lists,
      where each inner list represents a row of pixels and each pixel
      has a value of either 0 or 1.
    - largest_component_pixels (set): A set of pixel coordinates belonging
      to the largest connected component.

    Returns:
    - list: The modified binary image with non-largest component pixels zeroed
            out.
    """

    height = len(image)
    width = len(image[0])
    for i in range(height):
        for j in range(width):
            if (i, j) not in largest_component_pixels:
                image[i][j] = 0
    return image


def overall_score(data, background_size, model_py, check_samples=-1):
    """
    Computes the overall score based on SHAP values and image processing
    techniques.

    Args:
    - data (list): A list of input data.
    - background_size (int): The size of the background dataset used for SHAP
                        computation.
    - model_py: The model to explain.
    - check_samples (int, optional): The number of samples to check. Defaults
                                    to -1.

    Returns:
    - tuple: A tuple containing two arrays:
        - The mean number of pixels turned off across all samples.
        - The mean effect of pixel modifications across all samples.
    """
    ds = ImageDataset_normalised(data)
    device = torch.device('cpu')

    shap_loader = DataLoader(ds, batch_size=background_size, shuffle=True)
    background, _ = next(iter(shap_loader))
    background = background.to(device)

    explainer = shap.DeepExplainer(model_py, background)

    if check_samples == -1:
        samples_list = range(ds.__len__())
        len_samples = ds.__len__()
    else:
        samples_list = list(random.sample(range(0, ds.__len__()),
                                          check_samples))
        len_samples = check_samples

    pixels_off = []
    effect = []

    for sample_num in tqdm(range(len_samples)):
        image_input = ds.__getitem__(samples_list[sample_num])[0]

        test_image = np.expand_dims(image_input, axis=0)
        test_image = torch.tensor(test_image).to(device)
        shap_values = explainer.shap_values(test_image)

        if len(np.array(shap_values).shape) == 5:
            shap_numpy = np.array(shap_values).transpose(4, 0, 2, 3, 1)
        else:
            shap_numpy = np.array(shap_values).transpose(0, 2, 3, 1)
            shap_numpy = np.expand_dims(shap_numpy, axis=0)

        image_filtered = full_squeeze(image_input)
        _, upper = np.percentile(image_filtered, [2.5, 99.9])
        image_filtered[image_filtered < upper] = 0
        image_filtered[image_filtered != 0] = 1

        largest_component = connected_components(image_filtered)
        image_filtered = zero_non_largest_components(image_filtered,
                                                     largest_component)

        pixels_sample = []
        effect_sample = []
        for shap_values in shap_numpy:
            original = shap_values.copy()
            shap_filtered = full_squeeze(shap_values).copy()

            lower, upper = np.percentile(shap_filtered, [0.1, 99.9])

            shap_filtered[shap_filtered >= upper] = 1
            shap_filtered[shap_filtered <= lower] = 1
            shap_filtered[shap_filtered != 1] = 0

            mask = np.abs(shap_filtered - image_filtered)

            final_values = full_squeeze(original)
            final_values[mask == 0] = 0
            non_zero_pixels_effect = np.count_nonzero(final_values)
            sum_values = np.sum(np.abs(final_values))
            mean_effect = sum_values / non_zero_pixels_effect

            pixels_sample.append(non_zero_pixels_effect)
            effect_sample.append(mean_effect)

        pixels_off.append(pixels_sample)
        effect.append(effect_sample)

    return np.mean(pixels_off, axis=0), np.mean(effect, axis=0)


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
