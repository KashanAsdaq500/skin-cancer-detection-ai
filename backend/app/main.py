import os
import uuid
import base64
import json
from typing import Dict, List
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResult
)
from backend.app.model.labels import ISIC_CLASSES
from backend.app.model.pipeline import (
    run_genuine_or_triage_analysis,
    WEIGHTS_CHECKPOINT_PATH,
    ONNX_MODEL_PATH,
    DISCLAIMER_TEXT,
    CLINICAL_LIMITATIONS
)
from backend.app.training.train import PIPELINE_SPECIFICATION

app = FastAPI(
    title="Dermora AI Backend API",
    description="Clinical Quality Triage and Inference Service for Dermora AI",
    version="1.1.0-clinical"
)

DEFAULT_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

def get_cors_origins() -> List[str]:
    """
    Parses allowed CORS origins from the CORS_ORIGINS environment variable.
    Supports comma-separated strings or JSON arrays.
    
    Production configuration:
      Set the CORS_ORIGINS environment variable to your deployed frontend origin(s), e.g.:
        CORS_ORIGINS="https://dermora.vercel.app,https://dermora.example.com"
      or as a JSON array:
        CORS_ORIGINS='["https://dermora.vercel.app"]'

    Local development fallback:
      When CORS_ORIGINS is not set or empty, defaults safely to localhost origins:
      http://localhost:3000, http://127.0.0.1:3000, http://localhost:8000, http://127.0.0.1:8000.
    """
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_DEV_ORIGINS

    if raw == "*":
        return ["*"]

    origins: List[str] = []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                origins = [str(item).strip().rstrip("/") for item in parsed if str(item).strip()]
        except Exception:
            pass

    if not origins:
        origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]

    return origins if origins else DEFAULT_DEV_ORIGINS

# Configure CORS for Next.js frontend with production environment allowlist
ALLOWED_CORS_ORIGINS = get_cors_origins()
# Browser security specifications forbid allow_credentials=True when allow_origins is ["*"]
ALLOW_CREDENTIALS = ALLOWED_CORS_ORIGINS != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for analysis results (keyed by UUID)
ANALYSIS_STORE: Dict[str, PredictionResult] = {}

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint verifying backend service and genuine model availability."""
    has_model = ONNX_MODEL_PATH.exists() or WEIGHTS_CHECKPOINT_PATH.exists()
    return HealthResponse(
        status="operational",
        version="1.1.0-clinical",
        service="Dermora AI Clinical Vision Backend",
        timestamp=datetime.now(timezone.utc).isoformat(),
        is_genuine_model_active=has_model,
        disclaimer="Research & Educational Prototype Only (CC BY-NC 4.0)"
    )

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Returns honest model architecture, genuine weight status, dataset specifications and licence."""
    has_model = ONNX_MODEL_PATH.exists() or WEIGHTS_CHECKPOINT_PATH.exists()
    engine_name = "Trained Convolutional Neural Network (ONNX)" if has_model else "Dermoscopic Quality Triage & Input Rejection Validator (Awaiting HAM10000 Training)"
    
    return ModelInfoResponse(
        service_name="Dermora AI Diagnostic Vision Engine",
        version="1.1.0-clinical",
        is_genuine_model_active=has_model,
        is_genuine_clinical_weights=False,
        current_engine=engine_name,
        planned_architecture="PyTorch EfficientNet-B0 / ResNet34 with ONNX Runtime",
        target_dataset="HAM10000 / ISIC Archive (10,015 labeled dermoscopy images)",
        dataset_licence="CC BY-NC 4.0 (Non-Commercial Research & Educational Use Only)",
        classes=[{"code": c["code"], "name": c["name"], "risk": c["risk_level"]} for c in ISIC_CLASSES],
        disclaimer=DISCLAIMER_TEXT,
        limitations=CLINICAL_LIMITATIONS
    )

@app.get("/api/training-pipeline")
async def training_pipeline():
    """Provides the documented dataset training roadmap and hyperparameters."""
    return PIPELINE_SPECIFICATION

@app.post("/api/predict", response_model=PredictionResult)
async def predict_image(request: Request):
    """
    Accepts an uploaded image, evaluates dermoscopic quality and input validity,
    and runs genuine model inference if trained weights exist, or returns triage metrics.
    """
    content_type = request.headers.get("content-type", "").lower()
    image_bytes = None

    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")
            
        b64_str = data.get("image_base64") or data.get("image")
        if not b64_str:
            raise HTTPException(status_code=400, detail="Missing 'image_base64' field in JSON request.")
            
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
            
        try:
            image_bytes = base64.b64decode(b64_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image data.")
    else:
        # Accept raw binary payload (image/jpeg, image/png, application/octet-stream)
        image_bytes = await request.body()
        if image_bytes.startswith(b"{"):
            try:
                parsed = json.loads(image_bytes.decode("utf-8"))
                b64 = parsed.get("image_base64") or parsed.get("image")
                if b64:
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64)
            except Exception:
                pass

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file exceeds 15MB limit.")

    result_id = str(uuid.uuid4())
    try:
        result = run_genuine_or_triage_analysis(image_bytes, result_id=result_id)
        ANALYSIS_STORE[result_id] = result
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

@app.get("/api/results/{result_id}", response_model=PredictionResult)
async def get_result(result_id: str):
    """Fetches a previously processed analysis result by ID."""
    if result_id not in ANALYSIS_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis result with ID '{result_id}' was not found or has expired."
        )
    return ANALYSIS_STORE[result_id]
