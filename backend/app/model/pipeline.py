import io
import os
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from PIL import Image
from backend.app.model.labels import ISIC_CLASSES, CLASS_BY_CODE
from backend.app.schemas import ImageMetrics, ClassProbability, PredictionResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DISCLAIMER_TEXT = (
    "RESEARCH & EDUCATIONAL PROTOTYPE ONLY: Dermora AI is an experimental computational "
    "platform. It is NOT a medical device, is NOT FDA or CE approved, and DOES NOT "
    "provide medical diagnoses. Never rely on algorithmic outputs to evaluate skin lesions or "
    "defer professional medical care. Histopathological biopsy by a licensed physician remains "
    "the diagnostic gold standard."
)

CLINICAL_LIMITATIONS = [
    "Dermoscopy vs Phone-Camera: Models trained on polarized dermoscopy images cannot guarantee accuracy on non-dermoscopic ambient phone snapshots.",
    "Non-diagnostic output: Predictions are generated for research, algorithmic benchmarking, and educational evaluation only.",
    "Lighting & Artifact Sensitivity: Hair follicles, surgical markers, skin tone variations, ruler markings, and blur degrade vision metrics.",
    "Dataset Boundary: HAM10000 / ISIC datasets represent specific diagnostic categories and patient demographics; generalization to unseen populations requires clinical validation."
]

def resolve_onnx_model_path() -> Path:
    """
    Resolves the active ONNX model path.
    Prioritizes DERMORA_ONNX_PATH environment variable if specified,
    then the real trained artifacts directory (artifacts_real_training_20261020),
    then falls back to the default artifacts directory.
    """
    env_path = os.getenv("DERMORA_ONNX_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    candidates = [
        PROJECT_ROOT / "backend" / "app" / "training" / "artifacts_real_training_20261020" / "dermora_model.onnx",
        PROJECT_ROOT / "backend" / "app" / "training" / "artifacts" / "dermora_model.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]

def resolve_checkpoint_path() -> Path:
    """
    Resolves the active PyTorch checkpoint path.
    Prioritizes DERMORA_CHECKPOINT_PATH environment variable if specified,
    then the real trained artifacts directory (artifacts_real_training_20261020),
    then falls back to the default artifacts directory.
    """
    env_path = os.getenv("DERMORA_CHECKPOINT_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    candidates = [
        PROJECT_ROOT / "backend" / "app" / "training" / "artifacts_real_training_20261020" / "best_model.pt",
        PROJECT_ROOT / "backend" / "app" / "training" / "artifacts" / "best_model.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]

def has_genuine_model() -> bool:
    """Returns True if either the trained ONNX model or PyTorch weights exist."""
    return resolve_onnx_model_path().is_file() or resolve_checkpoint_path().is_file()

WEIGHTS_CHECKPOINT_PATH = resolve_checkpoint_path()
ONNX_MODEL_PATH = resolve_onnx_model_path()

def preprocess_for_eval(img: Image.Image) -> np.ndarray:
    """
    Reproduces the exact training evaluation preprocessing pipeline from train.py:
    1. Convert to RGB
    2. Resize to (256, 256) with Bilinear interpolation
    3. CenterCrop(224) -> (left=16, top=16, right=240, bottom=240)
    4. ToTensor: float32 scaling to [0.0, 1.0] and NCHW layout
    5. Normalize with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
    Returns shape (1, 3, 224, 224) float32 ndarray.
    """
    rgb_img = img.convert("RGB")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    resized = rgb_img.resize((256, 256), resample=resample)
    
    # Exact torchvision CenterCrop(224) from 256x256
    left = (256 - 224) // 2  # 16
    top = (256 - 224) // 2   # 16
    right = left + 224       # 240
    bottom = top + 224       # 240
    cropped = resized.crop((left, top, right, bottom))
    
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_arr = (arr - mean) / std
    
    # (H, W, C) -> (1, C, H, W)
    input_tensor = np.transpose(norm_arr, (2, 0, 1))[np.newaxis, ...]
    return input_tensor

def validate_dermoscopic_input(img: Image.Image) -> ImageMetrics:
    """
    Evaluates image quality, resolution, contrast, focus, and cutaneous chromaticity.
    Rejects low-resolution, blurred, washed-out, or non-skin/non-lesion photographs.
    """
    width, height = img.size
    aspect_ratio = round(width / max(height, 1), 2)
    
    # 1. Grayscale metrics
    gray = img.convert('L')
    gray_arr = np.asarray(gray, dtype=np.float32)
    mean_lum = float(np.mean(gray_arr))
    
    # Luminance-normalized gradient variance for sharpness evaluation:
    # Normalizing by local reference illumination ensures darker skin tones (Fitzpatrick IV-VI)
    # and low-key clinical exposures are not falsely flagged as blurred simply due to lower grayscale values.
    lum_scale = 128.0 / max(mean_lum, 25.0)
    norm_gray = gray_arr * lum_scale
    gy, gx = np.gradient(norm_gray)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = float(np.var(gnorm))
    sharpness_score = round(min((sharpness * 4.0), 100.0), 1)
    
    # Contrast
    contrast = float(np.std(gray_arr))
    contrast_score = round(min((contrast / 64.0) * 100.0, 100.0), 1)

    # 2. Cutaneous Chromaticity Ratio (HSV color space)
    # Calibrated across all Fitzpatrick phototypes (Types I through VI)
    # Covers fair, olive, warm tan, and deeply pigmented brown/black skin tones
    hsv_img = img.convert('HSV').resize((128, 128))
    hsv_arr = np.asarray(hsv_img, dtype=np.float32)
    h, s, v = hsv_arr[..., 0], hsv_arr[..., 1] / 255.0, hsv_arr[..., 2] / 255.0
    
    skin_mask = (
        ((h <= 38) | (h >= 220)) &
        (s >= 0.10) & (s <= 0.85) &
        (v >= 0.15) & (v <= 0.98)
    )
    skin_ratio = round(float(np.mean(skin_mask)), 3)

    rejections: List[str] = []
    notes: List[str] = []

    # Resolution check
    if width < 224 or height < 224:
        rejections.append(f"Image resolution too low ({width}x{height} px). Minimum required: 224x224 px.")
    else:
        notes.append(f"Resolution sufficient ({width}x{height} px)")

    # Aspect ratio check
    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        rejections.append("Unusual aspect ratio. Standard centered macro or dermoscopic framing is required.")

    # Illumination checks
    if mean_lum < 20.0:
        rejections.append("Severely underexposed photograph (too dark to resolve cutaneous structure).")
    elif mean_lum > 245.0:
        rejections.append("Severely overexposed photograph (blown-out highlights; surface detail lost).")
    else:
        notes.append(f"Mean luminance within acceptable clinical range ({round(mean_lum, 1)})")

    # Sharpness check (normalized gradient variance < 1.8 indicates true defocus/motion blur)
    if sharpness < 1.8:
        rejections.append("Severe blur detected. High-frequency lesion boundary gradients are lost.")
    elif sharpness_score < 20.0:
        notes.append("Moderate sharpness. Optical focus could be improved.")
    else:
        notes.append(f"Sharpness score optimal ({sharpness_score}/100)")

    # Cutaneous / Dermoscopy check
    if skin_ratio < 0.25:
        rejections.append(
            "Non-dermatological input detected. The image lacks expected skin/pigment chromaticity. "
            "Dermora AI is restricted to skin lesion photography and rejects graphics, documents, or non-skin objects."
        )
    else:
        notes.append(f"Cutaneous chromaticity verified ({int(skin_ratio*100)}% skin-tone profile)")

    is_supported = len(rejections) == 0
    status_label = "Input Supported for Dermoscopic Evaluation" if is_supported else "Input Rejected - Quality/Type Requirements Not Met"

    return ImageMetrics(
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        sharpness_score=sharpness_score,
        contrast_score=contrast_score,
        mean_luminance=round(mean_lum, 1),
        skin_tone_ratio=skin_ratio,
        is_supported_dermoscopic_input=is_supported,
        quality_status=status_label,
        rejection_reasons=rejections,
        quality_notes=notes
    )

def run_genuine_or_triage_analysis(
    image_bytes: bytes,
    result_id: str,
    include_preview: bool = True
) -> PredictionResult:
    """
    Executes image quality validation and input rejection.
    If verified model weights exist, runs real deep learning inference.
    If weights are not yet trained, honestly communicates pending training status without fake probabilities.
    """
    from datetime import datetime, timezone
    pil_img = Image.open(io.BytesIO(image_bytes))
    metrics = validate_dermoscopic_input(pil_img)

    # Check if genuine model weights exist
    active_onnx = resolve_onnx_model_path()
    active_pt = resolve_checkpoint_path()
    has_onnx = active_onnx.is_file()
    has_pt = active_pt.is_file()
    is_genuine_active = has_onnx or has_pt

    preview_data_url = None
    if include_preview:
        thumb = pil_img.copy()
        thumb.thumbnail((400, 400))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        preview_data_url = f"data:image/jpeg;base64,{b64}"

    # Scenario 1: Input rejected by quality / dermoscopy filter
    if not metrics.is_supported_dermoscopic_input:
        return PredictionResult(
            id=result_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_genuine_model_active=is_genuine_active,
            analysis_status="Input Rejected",
            triage_message="The uploaded image cannot be evaluated because it does not meet clinical dermoscopic quality or domain requirements.",
            is_supported_dermoscopic_input=False,
            rejection_reasons=metrics.rejection_reasons,
            predicted_class=None,
            predicted_code=None,
            confidence=None,
            risk_level=None,
            probabilities=[],
            image_metrics=metrics,
            model_version="Dermora-Input-Validator-v1.0",
            model_status="Input Rejection Filter Triggered",
            is_verified_clinical_model=False,
            training_dataset_target="HAM10000 / ISIC Archive",
            disclaimer=DISCLAIMER_TEXT,
            limitations=CLINICAL_LIMITATIONS,
            image_preview=preview_data_url
        )

    # Scenario 2: Input is valid, but model weights are not yet trained
    if not is_genuine_active:
        return PredictionResult(
            id=result_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_genuine_model_active=False,
            analysis_status="Quality Triage Passed — Model Training Pending",
            triage_message=(
                "Image passed dermoscopic quality and chromaticity validation. "
                "Genuine neural network inference is not yet active: model weights must be trained on "
                "the official HAM10000/ISIC dataset before disease classification and probability vectors can be displayed."
            ),
            is_supported_dermoscopic_input=True,
            rejection_reasons=[],
            predicted_class="Pending Genuine Training",
            predicted_code="PENDING",
            confidence=None,
            risk_level="Pending Model Weights",
            probabilities=[],
            image_metrics=metrics,
            model_version="Dermora-Quality-Triage-v1.0 (Weights Pending)",
            model_status="Awaiting Execution of PyTorch Training Pipeline on HAM10000 Dataset",
            is_verified_clinical_model=False,
            training_dataset_target="HAM10000 / ISIC Archive (10,015 images, CC BY-NC 4.0)",
            disclaimer=DISCLAIMER_TEXT,
            limitations=CLINICAL_LIMITATIONS,
            image_preview=preview_data_url
        )

    # Scenario 3: Genuine model weights are present -> Run genuine ONNX/PyTorch inference
    try:
        import onnxruntime as ort
        input_tensor = preprocess_for_eval(pil_img)

        session = ort.InferenceSession(str(active_onnx), providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        logits = session.run(None, ort_inputs)[0][0]

        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        from backend.app.training.dataset import INDEX_TO_LABEL, LABEL_DISPLAY_NAMES
        ranked_indices = np.argsort(probs)[::-1]
        top_idx = int(ranked_indices[0])
        top_code = INDEX_TO_LABEL[top_idx].upper()
        top_meta = CLASS_BY_CODE.get(top_code, {})
        top_name = top_meta.get("name", LABEL_DISPLAY_NAMES.get(INDEX_TO_LABEL[top_idx], top_code))
        top_risk = top_meta.get("risk_level", "Low Risk")
        top_conf = round(float(probs[top_idx]) * 100.0, 1)

        class_probs = []
        for idx in ranked_indices:
            code = INDEX_TO_LABEL[idx].upper()
            meta = CLASS_BY_CODE.get(code, {})
            name = meta.get("name", LABEL_DISPLAY_NAMES.get(INDEX_TO_LABEL[idx], code))
            r_level = meta.get("risk_level", "Low Risk")
            desc = meta.get("description", f"Predicted probability: {round(float(probs[idx]) * 100.0, 1)}%")
            class_probs.append(ClassProbability(
                class_name=name,
                code=code,
                probability=round(float(probs[idx]) * 100.0, 1),
                risk_level=r_level,
                clinical_description=desc
            ))

        return PredictionResult(
            id=result_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_genuine_model_active=True,
            analysis_status="Genuine Deep Learning Inference Completed",
            triage_message="Analysis generated by trained PyTorch/ONNX convolutional neural network on HAM10000 dataset.",
            is_supported_dermoscopic_input=True,
            rejection_reasons=[],
            predicted_class=top_name,
            predicted_code=top_code,
            confidence=top_conf,
            risk_level=top_risk,
            probabilities=class_probs,
            image_metrics=metrics,
            model_version="Dermora-EfficientNetB0-ONNX-v1.0",
            model_status="Active: Genuine trained model weights loaded",
            is_verified_clinical_model=False,
            training_dataset_target="HAM10000 / ISIC Archive (10,015 images, CC BY-NC 4.0)",
            disclaimer=DISCLAIMER_TEXT,
            limitations=CLINICAL_LIMITATIONS,
            image_preview=preview_data_url
        )
    except Exception as e:
        # Fallback if ONNX runtime fails
        return PredictionResult(
            id=result_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_genuine_model_active=False,
            analysis_status=f"Inference Error: {str(e)}",
            triage_message="Error loading or running ONNX inference session.",
            is_supported_dermoscopic_input=True,
            rejection_reasons=[str(e)],
            predicted_class=None,
            predicted_code=None,
            confidence=None,
            risk_level=None,
            probabilities=[],
            image_metrics=metrics,
            model_version="Error",
            model_status="Inference Failed",
            is_verified_clinical_model=False,
            training_dataset_target="HAM10000 / ISIC Archive",
            disclaimer=DISCLAIMER_TEXT,
            limitations=CLINICAL_LIMITATIONS,
            image_preview=preview_data_url
        )
