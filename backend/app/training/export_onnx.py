"""
Dermora AI - ONNX Model Export & Inference Parity Verification
Exports PyTorch weights to ONNX format and verifies prediction parity.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

def export_model_to_onnx(
    pytorch_model,
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
    opset_version: int = 17
) -> str:
    """
    Exports a trained PyTorch classifier to an optimized ONNX model.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for ONNX export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pytorch_model.eval()
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], requires_grad=False)

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_tensor"],
        output_names=["logits"],
        dynamic_axes={
            "input_tensor": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    return str(output_path)

def verify_onnx_parity(
    pytorch_model,
    onnx_path: str,
    input_size: Tuple[int, int] = (224, 224),
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Feeds identical random input tensors to both PyTorch and ONNX Runtime,
    verifying numerical output parity within the specified tolerance.
    """
    try:
        import torch
        import onnxruntime as ort
    except ImportError:
        return {
            "verified": False,
            "error": "PyTorch and onnxruntime are required for numerical parity verification."
        }

    pytorch_model.eval()
    dummy_input = torch.randn(2, 3, input_size[0], input_size[1])

    # 1. PyTorch forward pass
    with torch.no_grad():
        py_logits = pytorch_model(dummy_input).cpu().numpy()

    # 2. ONNX Runtime forward pass
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_logits = session.run(None, ort_inputs)[0]

    # 3. Compare maximum absolute error
    max_diff = float(np.max(np.abs(py_logits - ort_logits)))
    is_close = max_diff < tolerance

    return {
        "verified": is_close,
        "max_absolute_difference": max_diff,
        "tolerance": tolerance,
        "pytorch_output_shape": list(py_logits.shape),
        "onnx_output_shape": list(ort_logits.shape),
        "status": "PASS: ONNX and PyTorch outputs match within tolerance" if is_close else "FAIL: Output discrepancy detected"
    }
