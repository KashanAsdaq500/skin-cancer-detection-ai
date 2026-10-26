from typing import List, Dict, Optional, Any

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self):
            return {k: (v.dict() if hasattr(v, "dict") else [i.dict() if hasattr(i, "dict") else i for i in v] if isinstance(v, list) else v) for k, v in self.__dict__.items()}
        def model_dump(self):
            return self.dict()

class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    timestamp: str
    is_genuine_model_active: bool
    disclaimer: str

class ClassProbability(BaseModel):
    class_name: str
    code: str
    probability: float
    risk_level: str
    clinical_description: str

class ImageMetrics(BaseModel):
    width: int
    height: int
    aspect_ratio: float
    sharpness_score: float
    contrast_score: float
    mean_luminance: float
    skin_tone_ratio: float
    is_supported_dermoscopic_input: bool
    quality_status: str
    rejection_reasons: List[str]
    quality_notes: List[str]

class PredictionResult(BaseModel):
    id: str
    timestamp: str
    is_genuine_model_active: bool
    analysis_status: str
    triage_message: str
    is_supported_dermoscopic_input: bool
    rejection_reasons: List[str]
    predicted_class: Optional[str] = None
    predicted_code: Optional[str] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    probabilities: List[ClassProbability] = []
    image_metrics: ImageMetrics
    model_version: str
    model_status: str
    is_verified_clinical_model: bool
    training_dataset_target: str
    disclaimer: str
    limitations: List[str]
    image_preview: Optional[str] = None

class ModelInfoResponse(BaseModel):
    service_name: str
    version: str
    is_genuine_model_active: bool
    is_genuine_clinical_weights: bool
    current_engine: str
    planned_architecture: str
    target_dataset: str
    dataset_licence: str
    classes: List[Dict[str, str]]
    disclaimer: str
    limitations: List[str]
