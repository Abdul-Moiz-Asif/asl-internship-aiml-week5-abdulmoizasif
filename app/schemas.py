from pydantic import BaseModel, Field
from typing import list

class CancerPredictionInput(BaseModel):
    """
    Strict validation schema for the 30 features of the Breast Cancer dataset.
    Every field is required and strictly typed as a float.
    """
    # Mean Features
    mean_radius: float = Field(..., description="Mean radius of the tumor")
    mean_texture: float = Field(...)
    mean_perimeter: float = Field(...)
    mean_area: float = Field(...)
    mean_smoothness: float = Field(...)
    mean_compactness: float = Field(...)
    mean_concavity: float = Field(...)
    mean_concave_points: float = Field(...)
    mean_symmetry: float = Field(...)
    mean_fractal_dimension: float = Field(...)

    # Standard Error Features
    radius_error: float = Field(...)
    texture_error: float = Field(...)
    perimeter_error: float = Field(...)
    area_error: float = Field(...)
    smoothness_error: float = Field(...)
    compactness_error: float = Field(...)
    concavity_error: float = Field(...)
    concave_points_error: float = Field(...)
    symmetry_error: float = Field(...)
    fractal_dimension_error: float = Field(...)

    # Worst (Largest) Features
    worst_radius: float = Field(...)
    worst_texture: float = Field(...)
    worst_perimeter: float = Field(...)
    worst_area: float = Field(...)
    worst_smoothness: float = Field(...)
    worst_compactness: float = Field(...)
    worst_concavity: float = Field(...)
    worst_concave_points: float = Field(...)
    worst_symmetry: float = Field(...)
    worst_fractal_dimension: float = Field(...)

class BatchPredictionInput(BaseModel):
    """Bonus Feature: Accepts a batch (list) of multiple patient records"""
    records: list[CancerPredictionInput]

class PredictionResponse(BaseModel):
    """
    Strict output schema to standardize the API's response.
    Includes prediction, confidence probability, and model version.
    """
    prediction: str
    confidence_score: float
    model_version: str