import streamlit as st
import onnx
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import (load_models, model_details, sample_details,
                             conv2d_feature_vis_extra_layers,
                             conv2d_feature_vis_no_extra_layers,
                             find_components, MedPCacheDataset_normalised)
from xai_tools import (vision_shap, integrated_grad, deeplift,
                                 shap_overview)


def delete_torch_model_files():
    file_1 = os.path.join(str(Path(__file__).resolve().parent),
                          "TorchModel_torchviz")
    file_2 = os.path.join(str(Path(__file__).resolve().parent),
                          "TorchModel_torchviz.png")
    if os.path.exists(file_1):
        os.remove(file_1)
    if os.path.exists(file_2):
        os.remove(file_2)


st.set_page_config(page_title="Demo", layout="wide", page_icon="📈")

background = 40
samples = 10

st.title("XAI Dashboard")

uploaded_model = st.file_uploader("Upload model", type=["onnx"])
model_flag = False
if uploaded_model is not None:
    model = onnx.load(uploaded_model)
    model_py = load_models(model)
    model_flag = True

uploaded_data = st.file_uploader("Upload data", type=["h5"])
data_flag = False

if uploaded_data is not None:
    ds = MedPCacheDataset_normalised(uploaded_data)

if uploaded_model is not None and uploaded_data is not None:

    functionality = st.selectbox(
        "Choose the functionality",
        [
            "-Select-",
            "Model Details",
            "Data Sample Details",
            "Convolutional Features Isolated",
            "Convolutional Features Non - Isolated",
            "Sample Components",
            "Integrated Gradients",
            "Deep Lift",
            "SHAP single sample",
            "SHAP overview",
        ],
    )

    if functionality == "Model Details":
        st.header("Model Details")
        model_input, ground_truth = ds.__getitem__(0)
        sample_size = model_input.shape
        dot, model_summary = model_details(model_py, sample_size)
        dot = dot.render("TorchModel_torchviz", format="png")
        image_path = os.path.join(str(Path(__file__).resolve().parent),
                                  "TorchModel_torchviz.png")
        st.image(dot)
        # Display model summary
        st.subheader("Model Architecture:")
        st.text(model_summary)
    elif functionality == "Data Sample Details":
        delete_torch_model_files()
        st.header("Data Sample Details")
        # Ask for a number input
        sample_number = int(st.number_input("Sample Number:", step=1))
        if sample_number is not None:
            model_input, ground_truth = ds.__getitem__(sample_number)
            model_output = sample_details(model_py, model_input)
            arr = model_input.copy()
            while 1 in arr.shape:
                arr = np.squeeze(arr)

            plt.imshow(arr)
            title = "Model_output: " + str(model_output) + "\n"
            title = title + "Ground Truth: " + str(ground_truth)
            plt.title(title)
            st.pyplot(plt)

    elif functionality == "Convolutional Features Isolated":
        delete_torch_model_files()
        st.header("Convolutional Features Isolated")
        sample_number = int(st.number_input("Sample Number:", step=1))
        if sample_number is not None:
            model_input, ground_truth = ds.__getitem__(sample_number)

            arrays, names = conv2d_feature_vis_no_extra_layers(
                model_py, model_input)

            num_arrays = len(arrays)
            # Calculate the number of rows needed
            num_rows = (num_arrays + 1) // 2
            fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
            fig.tight_layout(pad=3.0)
            for i, array in enumerate(arrays):
                row = i // 2
                col = i % 2
                ax = axes[row, col] if num_rows > 1 else axes[col]
                im = ax.imshow(array, cmap="viridis")
                plt.colorbar(im, ax=ax)
                ax.set_title(names[i].split("(")[0] + str(i))
            st.pyplot(plt)

    elif functionality == "Convolutional Features Non - Isolated":
        delete_torch_model_files()
        st.header("Convolutional Features Non - Isolated")
        sample_number = int(st.number_input("Sample Number:", step=1))
        if sample_number is not None:
            model_input, ground_truth = ds.__getitem__(sample_number)

            arrays, names = conv2d_feature_vis_extra_layers(
                model_py, model_input)
            num_arrays = len(arrays)

            # Calculate the number of rows needed
            num_rows = (num_arrays + 1) // 2
            fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
            fig.tight_layout(pad=3.0)
            for i, array in enumerate(arrays):
                row = i // 2
                col = i % 2
                ax = axes[row, col] if num_rows > 1 else axes[col]
                im = ax.imshow(array, cmap="viridis")
                plt.colorbar(im, ax=ax)
                ax.set_title(names[i].split("(")[0] + str(i))
            st.pyplot(plt)

    elif functionality == "Sample Components":
        delete_torch_model_files()
        st.header("Sample Components")
        sample_number = int(st.number_input("Sample Number:", step=1))
        num_components = int(st.number_input("Number of Components:", step=1))
        if sample_number is not None and num_components is not None:
            if num_components < 2:
                st.write("No valid number for components")
            else:
                model_input, ground_truth = ds.__getitem__(sample_number)
                heatmaps = find_components(model_py, model_input,
                                           num_components)

                num_rows = (num_components +
                            1) // 2  # Calculate the number of rows needed
                fig, axes = plt.subplots(num_rows,
                                         2,
                                         figsize=(10, 5 * num_rows))
                fig.tight_layout(pad=3.0)

                for i in range(num_components):
                    row = i // 2
                    col = i % 2
                    ax = axes[row, col] if num_rows > 1 else axes[col]
                    ax.imshow(heatmaps[0][i])
                    ax.set_title("component_" + str(i))

                for i in range(num_components, num_rows * 2):
                    axes[num_rows - 1][i % 2].axis("off")

                plt.tight_layout()

                st.pyplot(plt)

    elif functionality == "Integrated Gradients":
        delete_torch_model_files()
        st.header("Integrated Gradients")
        sample_number = int(st.number_input("Sample Number:", step=1))
        if sample_number is not None:
            model_input, ground_truth = ds.__getitem__(sample_number)
            grads = integrated_grad(model_py, model_input)
            if len(grads) == 1:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))

                im = ax.imshow(grads)
                ax.axis("off")
                ax.set_title("integrated grad")
            else:

                # Calculate the number of rows needed
                num_rows = (len(grads) + 1) // 2
                fig, axes = plt.subplots(num_rows,
                                         2,
                                         figsize=(10, 5 * num_rows))
                fig.tight_layout(pad=3.0)

                for i, array in enumerate(grads):
                    row = i // 2
                    col = i % 2
                    ax = axes[row, col] if num_rows > 1 else axes[col]

                    im = ax.imshow(array)

                    ax.set_title("integrated grad value " + str(i))

                for i in range(len(grads), num_rows * 2):
                    axes[num_rows - 1][i % 2].axis("off")

                plt.tight_layout()
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            st.pyplot(plt)

    elif functionality == "Deep Lift":
        delete_torch_model_files()
        st.header("Deep Lift")
        sample_number = int(st.number_input("Sample Number:", step=1))
        if sample_number is not None:
            model_input, ground_truth = ds.__getitem__(sample_number)
            dl_arrays = deeplift(model_py, model_input)
            if len(dl_arrays) == 1:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))

                im = ax.imshow(dl_arrays[0])
                ax.axis("off")
                ax.set_title("integrated grad")
            else:

                num_rows = (len(dl_arrays) +
                            1) // 2  # Calculate the number of rows needed
                fig, axes = plt.subplots(num_rows,
                                         2,
                                         figsize=(10, 5 * num_rows))
                fig.tight_layout(pad=3.0)

                for i, array in enumerate(dl_arrays):
                    row = i // 2
                    col = i % 2
                    ax = axes[row, col] if num_rows > 1 else axes[col]

                    im = ax.imshow(array)

                    ax.set_title("integrated grad value " + str(i))

                for i in range(len(dl_arrays), num_rows * 2):
                    axes[num_rows - 1][i % 2].axis("off")

                plt.tight_layout()
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            st.pyplot(plt)

    elif functionality == "SHAP single sample":
        delete_torch_model_files()
        st.header("SHAP single sample")
        sample_number = int(st.number_input("Sample Number:", step=1))
        if sample_number is not None:
            model_input, ground_truth = ds.__getitem__(sample_number)
            plots = vision_shap(uploaded_data, samples, model_py, model_input)
            for i, plot in enumerate(plots):
                st.subheader("SHAP output for target " + str(i))
                st.pyplot(plot)

    elif functionality == "SHAP overview":
        delete_torch_model_files()
        st.header("SHAP overview")
        plots = shap_overview(uploaded_data, background, samples, model_py)
        for i, plot in enumerate(plots):
            st.subheader("SHAP pixels overview contribution for target " +
                         str(i))
            st.pyplot(plot)
