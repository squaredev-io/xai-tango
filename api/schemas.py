from pydantic import BaseModel, Field, root_validator
from typing import Literal, Optional

###### Banking schemas ######

class LimeRequest(BaseModel):
    row_index: int = Field(..., description="Row index of the data point to explain")

    @root_validator
    def validate_row_index(cls, values):
        row_index = values.get("row_index")
        if row_index < 0 or row_index >= len(X_test):
            raise ValueError(f"Row index must be between 0 and {len(X_test) - 1}")
        return values


class LimeResponse(BaseModel):
    plot_url: str


class ShapRequest(BaseModel):
    plot_type: Literal["summary", "bar", "waterfall", "heatmap", "beeswarm"] = Field(..., description="Type of SHAP plot", example="summary")
    data_point: Optional[int] = Field(None, description="Data point index for waterfall plot")

    @root_validator
    def validate_request(cls, values):
        plot_type = values.get("plot_type")
        data_point = values.get("data_point")
        
        if plot_type == "waterfall":
            if data_point is None:
                raise ValueError("For 'waterfall' plot_type, data_point must be specified.")
            if data_point < 0:
                raise ValueError(f"Data point must be a non-negative integer.")
        
        return values

class ShapResponse(BaseModel):
    plot_url: str
    title: str
    description: str
    where_it_helps: str
    how_to_use: str
    requirements: str


###### Vision Schemas #######

class ModelDetailsResponse(BaseModel):
    model_summary: str = Field(..., description="Summary of the model architecture.")
    architecture_diagram: str = Field(..., description="Base64-encoded image of the model architecture diagram.")


class SampleDetailsRequest(BaseModel):
    sample_index: int = Field(..., description="Index of the data sample to visualize.")

    @root_validator
    def validate_sample_index(cls, values):
        sample_index = values.get("sample_index")
        if sample_index < 0:
            raise ValueError("Sample index must be a non-negative integer.")
        return values


class SampleDetailsResponse(BaseModel):
    sample_image: str = Field(..., description="Base64-encoded image of the data sample visualization.")


class ConvolutionalFeaturesRequest(BaseModel):
    sample_index: int = Field(..., description="Index of the data sample to visualize convolutional features.")
    isolation: bool = Field(..., description="Flag indicating whether to isolate convolutional features (True) or not (False).")

    @root_validator
    def validate_sample_index(cls, values):
        sample_index = values.get("sample_index")
        if sample_index < 0:
            raise ValueError("Sample index must be a non-negative integer.")
        return values


class ConvolutionalFeaturesResponse(BaseModel):
    feature_maps: list[str] = Field(..., description="List of base64-encoded images of convolutional feature maps.")


class SampleComponentsRequest(BaseModel):
    sample_index: int = Field(..., description="Index of the data sample to visualize components.")
    num_components: int = Field(..., description="Number of independent components to extract (minimum: 2).")

    @root_validator
    def validate_num_components(cls, values):
        num_components = values.get("num_components")
        if num_components < 2:
            raise ValueError("Number of components must be at least 2.")
        return values


class SampleComponentsResponse(BaseModel):
    components: list[str] = Field(..., description="List of base64-encoded images of the sample components.")


class IntegratedGradientsRequest(BaseModel):
    sample_index: int = Field(..., description="Index of the data sample to analyze using Integrated Gradients.")

    @root_validator
    def validate_sample_index(cls, values):
        sample_index = values.get("sample_index")
        if sample_index < 0:
            raise ValueError("Sample index must be a non-negative integer.")
        return values


class IntegratedGradientsResponse(BaseModel):
    gradients: list[str] = Field(..., description="List of base64-encoded images of the Integrated Gradients visualization.")


class DeepLiftRequest(BaseModel):
    sample_index: int = Field(..., description="Index of the data sample to analyze using DeepLift.")

    @root_validator
    def validate_sample_index(cls, values):
        sample_index = values.get("sample_index")
        if sample_index < 0:
            raise ValueError("Sample index must be a non-negative integer.")
        return values


class DeepLiftResponse(BaseModel):
    deeplift_maps: list[str] = Field(..., description="List of base64-encoded images of the DeepLift visualization.")


class ShapSingleSampleRequest(BaseModel):
    sample_index: int = Field(..., description="Index of the data sample to analyze using SHAP.")

    @root_validator
    def validate_sample_index(cls, values):
        sample_index = values.get("sample_index")
        if sample_index < 0:
            raise ValueError("Sample index must be a non-negative integer.")
        return values


class ShapSingleSampleResponse(BaseModel):
    shap_plots: list[str] = Field(..., description="List of base64-encoded SHAP visualizations for the selected sample.")


class ShapOverviewResponse(BaseModel):
    overview_plots: list[str] = Field(..., description="List of base64-encoded SHAP overview visualizations.")
