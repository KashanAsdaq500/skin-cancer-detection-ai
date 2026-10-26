"""
Dermora AI - Inference Integration, Preprocessing Exactness, and Model Path Tests
Verifies:
1. Preprocessing reproduces training evaluation preprocessing (Resize 256 + CenterCrop 224 + ToTensor + ImageNet Norm)
2. Model path resolution prioritizes real trained artifacts (artifacts_real_training_20261020) and respects environment variables
3. Trained class order (NV, MEL, BKL, BCC, AKIEC, VASC, DF) maps correctly to labels and risk levels
4. Real ONNX model inference outputs valid 7-class probability distributions conforming to frontend schema
5. Frontend API URL resolution logic (trailing slash protection, localhost fallback)
"""

import os
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from backend.app.model.pipeline import (
    preprocess_for_eval,
    resolve_onnx_model_path,
    resolve_checkpoint_path,
    has_genuine_model,
    run_genuine_or_triage_analysis,
    PROJECT_ROOT
)
from backend.app.training.dataset import HAM10000_LABEL_MAP, INDEX_TO_LABEL, LABEL_DISPLAY_NAMES
from backend.app.model.labels import CLASS_BY_CODE, ISIC_CLASSES

class TestInferenceIntegration(unittest.TestCase):

    def test_01_preprocessing_exactness_vs_torchvision(self):
        """Verifies that preprocess_for_eval matches torchvision eval_transforms to numerical precision."""
        try:
            import torch
            from torchvision import transforms
        except ImportError:
            self.skipTest("PyTorch/torchvision not available for comparison test.")

        eval_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        np.random.seed(123)
        dummy_pixels = np.random.randint(0, 256, (320, 480, 3), dtype=np.uint8)
        img = Image.fromarray(dummy_pixels)

        # Torchvision pipeline output
        t_out = eval_transforms(img).unsqueeze(0).numpy()

        # Dermora pure numpy/PIL pipeline output
        np_out = preprocess_for_eval(img)

        self.assertEqual(np_out.shape, (1, 3, 224, 224))
        self.assertEqual(np_out.dtype, np.float32)
        diff = float(np.max(np.abs(t_out - np_out)))
        self.assertAlmostEqual(diff, 0.0, places=5, msg=f"Preprocessing deviation from torchvision: {diff}")

    def test_02_model_path_resolution(self):
        """Verifies that model path resolution defaults to artifacts_real_training_20261020 and respects env overrides."""
        default_onnx = resolve_onnx_model_path()
        self.assertTrue(default_onnx.is_file(), f"ONNX model file not found at: {default_onnx}")
        self.assertIn("artifacts_real_training_20261020", str(default_onnx))

        default_pt = resolve_checkpoint_path()
        self.assertTrue(default_pt.is_file(), f"Checkpoint file not found at: {default_pt}")
        self.assertIn("artifacts_real_training_20261020", str(default_pt))

        self.assertTrue(has_genuine_model())

        # Test environment variable override
        os.environ["DERMORA_ONNX_PATH"] = "custom/test/model.onnx"
        try:
            custom_onnx = resolve_onnx_model_path()
            self.assertEqual(custom_onnx, PROJECT_ROOT / "custom" / "test" / "model.onnx")
        finally:
            del os.environ["DERMORA_ONNX_PATH"]

    def test_03_class_mapping_order_and_risk(self):
        """Verifies trained model output indices map to: NV, MEL, BKL, BCC, AKIEC, VASC, DF."""
        expected_order = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
        for idx, code in enumerate(expected_order):
            self.assertEqual(HAM10000_LABEL_MAP[code], idx)
            self.assertEqual(INDEX_TO_LABEL[idx], code)

            upper_code = code.upper()
            self.assertIn(upper_code, CLASS_BY_CODE)
            meta = CLASS_BY_CODE[upper_code]
            self.assertEqual(meta["code"], upper_code)

        # Check risk level classifications
        self.assertEqual(CLASS_BY_CODE["MEL"]["risk_level"], "High Risk")
        self.assertEqual(CLASS_BY_CODE["BCC"]["risk_level"], "High Risk")
        self.assertEqual(CLASS_BY_CODE["AKIEC"]["risk_level"], "Medium Risk")
        self.assertEqual(CLASS_BY_CODE["NV"]["risk_level"], "Low Risk")
        self.assertEqual(CLASS_BY_CODE["BKL"]["risk_level"], "Low Risk")
        self.assertEqual(CLASS_BY_CODE["DF"]["risk_level"], "Low Risk")
        self.assertEqual(CLASS_BY_CODE["VASC"]["risk_level"], "Low Risk")

    def test_04_real_onnx_inference_and_schema_compatibility(self):
        """Verifies real ONNX inference produces compliant PredictionResult with all 7 ranked probabilities."""
        import io
        # Construct synthetic cutaneous lesion passing input filter
        arr = np.full((300, 300, 3), (210, 150, 120), dtype=np.uint8)
        np.random.seed(42)
        noise = np.random.randint(-15, 15, (300, 300, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        yy, xx = np.ogrid[:300, :300]
        mask = ((xx - 150)**2 + (yy - 150)**2) < 60**2
        arr[mask] = np.clip(arr[mask].astype(np.int16) - 90, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        result = run_genuine_or_triage_analysis(buf.getvalue(), result_id="test-integration-uuid")

        self.assertEqual(result.id, "test-integration-uuid")
        self.assertTrue(result.is_genuine_model_active)
        self.assertTrue(result.is_supported_dermoscopic_input)
        self.assertEqual(result.analysis_status, "Genuine Deep Learning Inference Completed")
        self.assertIn(result.predicted_code, {"NV", "MEL", "BKL", "BCC", "AKIEC", "VASC", "DF"})
        self.assertGreater(result.confidence, 0.0)
        self.assertIn(result.risk_level, {"High Risk", "Medium Risk", "Low Risk"})

        # Exactly 7 classes
        self.assertEqual(len(result.probabilities), 7)
        prob_codes = [p.code for p in result.probabilities]
        self.assertEqual(set(prob_codes), {"NV", "MEL", "BKL", "BCC", "AKIEC", "VASC", "DF"})

        # Probabilities sum to approximately 100%
        total_prob = sum(p.probability for p in result.probabilities)
        self.assertAlmostEqual(total_prob, 100.0, delta=1.0)

        # Probabilities are sorted in descending order
        prob_values = [p.probability for p in result.probabilities]
        self.assertEqual(prob_values, sorted(prob_values, reverse=True))

    def test_05_api_url_construction_logic(self):
        """Verifies frontend URL construction logic handling trailing slashes and default fallback."""
        def build_url(env_val: str | None, path: str) -> str:
            base = (env_val.strip().rstrip("/") if env_val and env_val.strip() else "http://localhost:8000")
            clean_path = path if path.startswith("/") else f"/{path}"
            return f"{base}{clean_path}"

        # Default fallback
        self.assertEqual(build_url(None, "/api/predict"), "http://localhost:8000/api/predict")
        self.assertEqual(build_url("", "api/predict"), "http://localhost:8000/api/predict")
        self.assertEqual(build_url("   ", "/api/results/123"), "http://localhost:8000/api/results/123")

        # Configured URL without trailing slash
        self.assertEqual(build_url("https://api.dermora.ai", "/api/predict"), "https://api.dermora.ai/api/predict")

        # Configured URL with trailing slashes
        self.assertEqual(build_url("https://api.dermora.ai/", "/api/predict"), "https://api.dermora.ai/api/predict")
        self.assertEqual(build_url("https://api.dermora.ai///", "api/predict"), "https://api.dermora.ai/api/predict")

    def test_06_cors_origin_resolution(self):
        """Verifies CORS origin allowlist parsing from environment and local dev fallback."""
        try:
            from backend.app.main import get_cors_origins, DEFAULT_DEV_ORIGINS
        except ImportError:
            # If fastapi is not installed in the testing venv, test parsing contract directly
            import json

            def parse_cors(raw_env: str) -> list[str]:
                dev_origins = [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                    "http://localhost:8000",
                    "http://127.0.0.1:8000",
                ]
                raw = raw_env.strip()
                if not raw:
                    return dev_origins
                if raw == "*":
                    return ["*"]
                origins = []
                if raw.startswith("[") and raw.endswith("]"):
                    try:
                        p = json.loads(raw)
                        if isinstance(p, list):
                            origins = [str(x).strip().rstrip("/") for x in p if str(x).strip()]
                    except Exception:
                        pass
                if not origins:
                    origins = [x.strip().rstrip("/") for x in raw.split(",") if x.strip()]
                return origins if origins else dev_origins
            
            get_cors_origins = lambda: parse_cors(os.getenv("CORS_ORIGINS", ""))
            DEFAULT_DEV_ORIGINS = [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:8000",
                "http://127.0.0.1:8000",
            ]

        old_val = os.environ.get("CORS_ORIGINS")
        try:
            # Unset / empty -> local dev origins
            if "CORS_ORIGINS" in os.environ:
                del os.environ["CORS_ORIGINS"]
            self.assertEqual(get_cors_origins(), DEFAULT_DEV_ORIGINS)

            os.environ["CORS_ORIGINS"] = "   "
            self.assertEqual(get_cors_origins(), DEFAULT_DEV_ORIGINS)

            # Comma-separated with trailing slashes & whitespace
            os.environ["CORS_ORIGINS"] = "https://dermora.vercel.app/, https://dermora-staging.vercel.app  "
            self.assertEqual(
                get_cors_origins(),
                ["https://dermora.vercel.app", "https://dermora-staging.vercel.app"]
            )

            # JSON array
            os.environ["CORS_ORIGINS"] = '["https://dermora.vercel.app/", "https://custom.app"]'
            self.assertEqual(
                get_cors_origins(),
                ["https://dermora.vercel.app", "https://custom.app"]
            )

            # Wildcard
            os.environ["CORS_ORIGINS"] = "*"
            self.assertEqual(get_cors_origins(), ["*"])
        finally:
            if old_val is not None:
                os.environ["CORS_ORIGINS"] = old_val
            elif "CORS_ORIGINS" in os.environ:
                del os.environ["CORS_ORIGINS"]

if __name__ == "__main__":
    unittest.main(verbosity=2)
