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

# Agg backend for non-GUI rendering
matplotlib.use("Agg")  
app = FastAPI()

# Preloaded model and dataset paths
MODEL_PATH = "xai_banking/utils/random_forest_model.pkl"
DATA_PATH = "xai_banking/utils/synthetic_dataset.csv"

# Load model and dataset at startup
model = None
processed_data = None
X_train, X_test, y_train, y_test = None, None, None, None


@app.on_event("startup")
def load_resources():
    global model, processed_data, X_train, X_test, y_train, y_test
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        raise RuntimeError("Model or dataset not found.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    raw_data = pd.read_csv(DATA_PATH)
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
        model=model,
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
    shap_values = shap_explainer(model, X_test)

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
