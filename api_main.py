from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, root_validator, ValidationError
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from xai_banking.utils.data_processing import preprocess_data
from xai_banking.utils.explainers import lime_explainer, shap_explainer
import shap
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
from api.schemas import *
from xaivision.utils import *
from xaivision.xai_tools import *

# Agg backend for non-GUI rendering
matplotlib.use("Agg")  
app = FastAPI()

# Preloaded banking_model and dataset paths
BANKING_MODEL_PATH = "xai_banking/utils/random_forest_model.pkl"
BANKING_DATA_PATH = "xai_banking/utils/synthetic_dataset.csv"

MODEL_PATH = "xaivision/model.onnx"
DATA_PATH = "xaivision/data1.h5"


# Load banking_model and dataset at startup
banking_model = None
processed_data = None
model = None
dataset = None
X_train, X_test, y_train, y_test = None, None, None, None


@app.on_event("startup")
def load_resources():
    global banking_model, processed_data, X_train, X_test, y_train, y_test, model, dataset
    if not os.path.exists(BANKING_MODEL_PATH) or not os.path.exists(BANKING_DATA_PATH):
        raise RuntimeError("Banking Model or dataset not found.")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        raise RuntimeError("Vision Model or dataset not found.")

    with open(BANKING_MODEL_PATH, "rb") as f:
        banking_model = pickle.load(f)

    model = load_models(MODEL_PATH)
    dataset = preprocess_dataset(DATA_PATH)

    raw_data = pd.read_csv(BANKING_DATA_PATH)
    processed_data = preprocess_data(raw_data)

    X = processed_data.drop(columns=["label_fraud_post"])
    y = processed_data["label_fraud_post"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


class LimeRequest(BaseModel):
    row_index: int = Field(..., description="Row index of the data point to explain")

    @root_validator
    def validate_row_index(cls, values):
        row_index = values.get("row_index")
        if row_index < 0 or row_index >= len(X_test):
            raise ValueError(f"Row index must be between 0 and {len(X_test) - 1}")
        return values


class ShapRequest(BaseModel):
    plot_type: str = Field(..., description="Type of SHAP plot", example="summary")
    data_point: int = Field(None, description="Data point index for waterfall plot")

    @root_validator
    def validate_request(cls, values):
        plot_type = values.get("plot_type")
        data_point = values.get("data_point")
        if plot_type == "waterfall":
            if data_point is None:
                raise ValueError("For 'waterfall' plot_type, data_point must be specified.")
            if data_point < 0 or data_point >= len(X_test):
                raise ValueError(f"Data point must be between 0 and {len(X_test) - 1}")
        return values


@app.get("/", tags=["Banking"])
def root():
    return {"message": "Welcome to the XAI API. Use /lime or /shap endpoints for explanations."}


@app.post("/lime/", tags=["Banking"])
def lime_explanation(request: LimeRequest):
    """
    Generate LIME explanation for a specific row in the test dataset.
    """
    # Generate LIME explanation
    explanation = lime_explainer(
        banking_model=banking_model,
        X_train=X_train,
        X_test=X_test,
        selected_row_index=request.row_index,
        class_names=["Not Fraud", "Fraud"]
    )

    # Save explanation plot as bytes
    fig = explanation.as_pyplot_figure()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    # Encode image in base64
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"plot_base64": image_base64}


@app.post("/shap/", tags=["Banking"])
def shap_explanation(request: ShapRequest):
    """
    Generate SHAP explanation for the test dataset.
    Supported plot types: 'summary', 'bar', 'waterfall', 'heatmap', 'beeswarm'.
    """
    # Generate SHAP values
    shap_values = shap_explainer(banking_model, X_test)

    # Save plots based on the requested type
    buf = BytesIO()
    plt.clf()
    plt.figure(figsize=(12, 8))

    if request.plot_type == "summary":
        shap.summary_plot(shap_values, X_test, show=False)
    elif request.plot_type == "bar":
        shap.plots.bar(shap_values)
    elif request.plot_type == "waterfall":
        if request.data_point is None:
            raise HTTPException(status_code=400, detail="Data point must be specified for waterfall plot.")
        shap.plots.waterfall(shap_values[request.data_point])
    elif request.plot_type == "heatmap":
        shap.plots.heatmap(shap_values)
    elif request.plot_type == "beeswarm":
        shap.plots.beeswarm(shap_values)

    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Encode image in base64
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"plot_base64": image_base64}

@app.get("/", tags=["Vision"])
def root():
    return {"message": "Welcome to the Vision XAI API. Use specific endpoints for visual explanations."}

@app.get("/model-details/", tags=["Vision"], response_model=ModelDetailsResponse)
def get_model_details():
    """
    Retrieve model architecture and details.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if dataset is None or len(dataset) == 0:
        raise HTTPException(status_code=500, detail="Dataset not loaded or is empty.")

    # Fetch the first sample
    try:
        sample_input = dataset[0]  # Use directly since it's not a tuple
        sample_size = sample_input.shape
        dot, summary = model_details(model, sample_size)
        dot_path = "model_architecture.png"
        dot.render(dot_path, format="png")
        with open(dot_path, "rb") as f:
            architecture_diagram = base64.b64encode(f.read()).decode("utf-8")
        return ModelDetailsResponse(model_summary=summary, architecture_diagram=architecture_diagram)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing dataset: {str(e)}")

@app.post("/sample-details/", tags=["Vision"])
def get_sample_details(request: SampleDetailsRequest):
    """
    Retrieve details of a specific data sample.
    """
    try:
        # Fetch the sample from the dataset
        sample_input = dataset[request.sample_index]  # Assuming dataset returns a single array
        
        # Extract ground_truth if it's embedded or known separately
        # Placeholder ground_truth if not explicitly provided
        ground_truth = "Unknown"  # Replace this with actual logic

        # Generate model output
        model_output = sample_details(model, sample_input)

        # Visualize the input sample
        plt.clf()
        plt.imshow(np.squeeze(sample_input))
        plt.title(f"Model Output: {model_output}\nGround Truth: {ground_truth}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Encode image in base64
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return SampleDetailsResponse(
            model_output=model_output,
            ground_truth=ground_truth,
            sample_image=image_base64  # This ensures the required field is populated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sample details: {str(e)}")


@app.post("/convolution-features/", tags=["Vision"])
def get_convolution_features(request: ConvolutionalFeaturesRequest):
    """
    Generate convolutional feature visualizations for a specific sample.
    """
    try:
        # Extract the sample from the dataset using sample_index from the request
        sample_data = dataset[request.sample_index]

        # If the dataset returns a tuple, take the first element as input
        if isinstance(sample_data, tuple):
            sample_input = sample_data[0]
        else:
            sample_input = sample_data

        # Dynamically determine reshape dimensions for a square input
        total_elements = sample_input.size
        channels = 1  # Assuming grayscale for simplicity
        height = int(total_elements ** 0.5)  # Assume square
        width = int(total_elements / height)

        if height * width != total_elements:
            raise ValueError(
                f"Cannot reshape array of size {total_elements} into a valid image shape."
            )

        # Reshape to 4D: [batch_size, channels, height, width]
        sample_input = np.reshape(sample_input, (1, channels, height, width))

        # Generate feature visualizations based on the isolation flag
        if request.isolation:
            arrays, names = conv2d_feature_vis_no_extra_layers(model, sample_input[0])  # Pass unbatched input
        else:
            arrays, names = conv2d_feature_vis_extra_layers(model, sample_input[0])  # Pass unbatched input

        # Create base64-encoded images for the feature maps
        feature_maps = []
        for array, name in zip(arrays, names):
            fig, ax = plt.subplots()
            im = ax.imshow(array, cmap="viridis")
            plt.colorbar(im, ax=ax)
            ax.set_title(name.split("(")[0])
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            feature_maps.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close(fig)

        return {"feature_maps": feature_maps}

    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid sample index provided.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing convolution features: {str(e)}")

@app.post("/sample-components/", tags=["Vision"], response_model=SampleComponentsResponse)
def get_sample_components(request: SampleComponentsRequest):
    """
    Visualize independent components for a specific sample.
    """
    try:
        # Validate number of components
        if request.num_components < 2:
            raise HTTPException(
                status_code=400, detail="Number of components must be at least 2."
            )

        # Fetch sample data
        sample_data = dataset[request.sample_index]
        if isinstance(sample_data, tuple):
            model_input = sample_data[0]  # Input data
        else:
            model_input = sample_data  # Single input without ground truth

        # Log original data for debugging
        print(f"Original model_input shape: {model_input.shape}")

        # Reshape input to 4D [B, C, H, W]
        if model_input.ndim == 2:  # [H, W]
            model_input = model_input[None, None, :, :]  # Add batch and channel dims
        elif model_input.ndim == 3:  # [C, H, W]
            model_input = model_input[None, :, :, :]  # Add batch dim
        elif model_input.ndim == 4:  # Already valid [B, C, H, W]
            pass
        else:
            raise ValueError(f"Unexpected input shape: {model_input.shape}")

        # Final validation and log reshaped data
        print(f"Reshaped model_input shape: {model_input.shape}")
        model_input = model_input.astype(np.float32)  # Convert to float32

        # Ensure input is unbatched if needed
        unbatched_input = model_input.squeeze(0)  # Remove batch dimension
        print(f"Unbatched model_input shape: {unbatched_input.shape}")

        # Generate heatmaps using the find_components function
        heatmaps = find_components(model, unbatched_input, request.num_components)
        print(f"Heatmaps shape: {len(heatmaps)}")

        # Create a list of base64-encoded images
        images = []
        for i, heatmap in enumerate(heatmaps[0]):
            plt.imshow(heatmap, cmap="viridis")
            plt.title(f"Component {i}")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()

            # Convert to base64
            images.append(base64.b64encode(buf.read()).decode("utf-8"))

        # Return the heatmaps as base64-encoded images
        return SampleComponentsResponse(components=images)

    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid sample index.")
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing sample components: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing sample components: {str(e)}"
        )

@app.post("/integrated-gradients/", tags=["Vision"], response_model=IntegratedGradientsResponse)
def get_integrated_gradients(request: IntegratedGradientsRequest):
    """
    Generate Integrated Gradients visualizations.
    """
    try:
        # Fetch sample data
        sample_data = dataset[request.sample_index]
        model_input = sample_data[0] if isinstance(sample_data, tuple) else sample_data

        # Log original shape
        print(f"Original model_input shape: {model_input.shape}")

        # Reshape input to [1, 1, H, W] if needed
        if model_input.ndim == 2:  # [H, W]
            model_input = model_input[None, None, :, :]
        elif model_input.ndim == 3:  # [C, H, W]
            model_input = model_input[None, :, :, :]
        elif model_input.ndim == 4:  # Already valid [B, C, H, W]
            pass
        else:
            raise ValueError(f"Unexpected input shape: {model_input.shape}")

        # Log reshaped input
        print(f"Reshaped model_input shape: {model_input.shape}")

        # Compute Integrated Gradients
        attributions = integrated_grad(model, model_input.squeeze(0))  # Remove batch dim

        # Create heatmap visualizations
        images = []
        for attr in attributions:
            plt.imshow(attr.squeeze(), cmap="viridis")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()
            images.append(base64.b64encode(buf.read()).decode("utf-8"))

        return IntegratedGradientsResponse(gradients=images)

    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid sample index.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error processing integrated gradients: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error processing integrated gradients: {str(e)}")

@app.post("/deep-lift/", tags=["Vision"], response_model=DeepLiftResponse)
def get_deep_lift(request: DeepLiftRequest):
    """
    Generate DeepLift visualizations.
    """
    try:
        # Fetch sample data
        sample_data = dataset[request.sample_index]

        # Handle dataset structure
        if isinstance(sample_data, tuple):
            sample_input = sample_data[0]  # Input data only
        else:
            sample_input = sample_data

        # Log original shape
        print(f"Original sample_input shape: {sample_input.shape}")

        # Reshape sample_input to ensure it has a single channel
        if sample_input.ndim == 2:  # [H, W]
            reshaped_input = np.expand_dims(sample_input, axis=0)  # [1, H, W]
        elif sample_input.ndim == 3:  # [C, H, W]
            if sample_input.shape[0] == 1:
                reshaped_input = sample_input  # Already has one channel
            else:
                reshaped_input = sample_input[:1]  # Take the first channel
        elif sample_input.ndim == 4:  # [B, C, H, W]
            reshaped_input = sample_input[0, :1, :, :]  # First batch, first channel
        else:
            raise ValueError(f"Unexpected input shape: {sample_input.shape}")

        # Log reshaped data
        print(f"Reshaped sample_input shape for DeepLift: {reshaped_input.shape}")

        # Ensure numpy float32 format
        reshaped_input = reshaped_input.astype(np.float32)

        # Call DeepLift function
        dl_arrays = deeplift(model, reshaped_input)

        # Generate visualizations
        images = []
        for dl_array in dl_arrays:
            plt.imshow(dl_array, cmap="viridis")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()
            images.append(base64.b64encode(buf.read()).decode("utf-8"))

        # Return DeepLift visualizations
        return DeepLiftResponse(deeplift_maps=images)

    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid sample index.")
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing DeepLift: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing DeepLift: {str(e)}"
        )

@app.post("/shap-single/", tags=["Vision"], response_model=ShapSingleSampleResponse)
def get_shap_single_sample(request: ShapSingleSampleRequest):
    """
    Generate SHAP visualizations for a single sample.
    """
    try:
        # Fetch sample data
        sample_data = dataset[request.sample_index]
        if isinstance(sample_data, tuple):
            # Unpack as many values as returned by the dataset
            sample_input, ground_truth, *rest = sample_data
        else:
            sample_input = sample_data  # Dataset returns only input data

        # Ensure sample_input has shape [channels, height, width]
        if sample_input.ndim == 2:  # [height, width]
            # Expand dimensions to add the channel dimension
            sample_input = np.expand_dims(sample_input, axis=0)  # [1, height, width]
        elif sample_input.ndim == 3:
            # If it already has 3 dimensions, assume it's [channels, height, width]
            pass
        else:
            raise ValueError(f"Unexpected sample_input shape: {sample_input.shape}")

        # Call SHAP function and capture both return values
        plots, shap_numpy = vision_shap(DATA_PATH, 10, model, sample_input)

        # Convert plots to images
        images = []
        for plot in plots:
            buf = BytesIO()
            plot.savefig(buf, format="png")
            buf.seek(0)
            plt.close(plot)
            images.append(base64.b64encode(buf.read()).decode("utf-8"))

        # Return SHAP visualizations
        return ShapSingleSampleResponse(shap_plots=images)

    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid sample index.")
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing SHAP: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing SHAP: {str(e)}"
        )


@app.get("/shap-overview/", tags=["Vision"], response_model=ShapOverviewResponse)
def get_shap_overview():
    """
    Generate SHAP overview visualizations.
    """
    plots = shap_overview(DATA_PATH, 40, 10, model)
    images = []
    for plot in plots:
        buf = BytesIO()
        plot.savefig(buf, format="png")
        buf.seek(0)
        plt.close(plot)
        images.append(base64.b64encode(buf.read()).decode("utf-8"))
    return ShapOverviewResponse(overview_plots=images)