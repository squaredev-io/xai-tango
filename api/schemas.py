from pydantic import BaseModel
from typing import Literal


class LimeRequest(BaseModel):
    """
    Schema for LIME explanation request.
    """
    row_index: int


class LimeResponse(BaseModel):
    """
    Schema for LIME explanation response.
    """
    plot_url: str


class ShapRequest(BaseModel):
    """
    Schema for SHAP explanation request.
    """
    plot_type: Literal["summary", "bar", "waterfall", "heatmap", "beeswarm"]


class ShapResponse(BaseModel):
    """
    Schema for SHAP explanation response.
    """
    plot_url: str
    title: str
    description: str
    where_it_helps: str
    how_to_use: str
    requirements: str
