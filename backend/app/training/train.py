"""
Dermora AI - End-to-End PyTorch Training, Validation, and Evaluation Pipeline
Target Dataset: HAM10000 / ISIC 2018 Task 3
Licence: CC BY-NC 4.0 (Non-Commercial Research & Educational Use Only)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from backend.app.training.dataset import (
    verify_dataset_directory,
    create_stratified_group_splits,
    check_group_leakage,
    HAM10000TorchDataset,
    HAM10000_LABEL_MAP
)
from backend.app.training.model import build_classifier_model
from backend.app.training.loss import compute_balanced_class_weights, build_loss_criterion
from backend.app.training.evaluate import evaluate_predictions
from backend.app.training.export_onnx import export_model_to_onnx, verify_onnx_parity

PIPELINE_SPECIFICATION: Dict[str, Any] = {
    "project": "Dermora AI",
    "dataset": {
        "name": "HAM10000 / ISIC 2018 Task 3",
        "samples": 10015,
        "classes": list(HAM10000_LABEL_MAP.keys()),
        "licence": "CC BY-NC 4.0 (Non-commercial research only)",
        "leakage_protection": "Stratified grouped partition on 'lesion_id' to prevent image-angle overlap"
    },
    "model_options": ["efficientnet_b0", "resnet34"],
    "loss_options": ["weighted_ce", "focal"],
    "export_format": "ONNX (opset 17)"
}

def set_reproducible_seed(seed: int = 42):
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def save_checkpoint_atomic(state_dict: Dict[str, Any], target_path: Path) -> Path:
    """
    Safely writes checkpoint to a temporary file first, then atomically replaces
    the target file to prevent corrupted or partial checkpoints on interruption.
    """
    import torch
    target_path = Path(target_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.stem}.tmp_{os.getpid()}_{int(time.time()*1000)}{target_path.suffix}")
    try:
        torch.save(state_dict, temp_path)
        os.replace(temp_path, target_path)
        return target_path
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

def inspect_and_load_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device=None,
    expected_backbone: Optional[str] = None,
    log_func=print
) -> Dict[str, Any]:
    """
    Loads model weights and training states from a checkpoint.
    Verifies availability of optimizer, scaler, scheduler, and early-stopping states.
    If full state is present, configures exact-state resume.
    If states are missing (e.g. legacy checkpoint), explicitly warns and configures
    a weights-only warm start without fabricating missing optimizer/scheduler states.
    """
    import torch
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    log_func(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

    ckpt_backbone = checkpoint.get("backbone")
    if expected_backbone and ckpt_backbone and ckpt_backbone != expected_backbone:
        raise ValueError(
            f"Checkpoint backbone '{ckpt_backbone}' does not match requested pipeline backbone '{expected_backbone}'."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint '{checkpoint_path.name}' is missing mandatory key 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    log_func(f"  • Model weights ({len(checkpoint['model_state_dict'])} tensors) loaded into {expected_backbone or ckpt_backbone or 'model'}.")

    has_opt = ("optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None)
    has_sched = ("scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None)
    has_early_stop = ("epochs_no_improve" in checkpoint and ("best_val_bma" in checkpoint or "val_bma" in checkpoint))

    if scaler is not None:
        has_scaler = ("scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None)
    else:
        # On CPU or when AMP is disabled, scaler_state_dict is present if the key exists in checkpoint
        has_scaler = ("scaler_state_dict" in checkpoint)

    missing_states = []
    if not has_opt: missing_states.append("optimizer_state_dict")
    if not has_scaler: missing_states.append("scaler_state_dict")
    if not has_sched: missing_states.append("scheduler_state_dict")
    if not has_early_stop: missing_states.append("early_stopping_state")

    # 1. Optimizer state
    if optimizer is not None:
        if has_opt:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if device is not None and getattr(device, "type", None) == "cuda":
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            log_func("  • Optimizer state restored successfully.")
        else:
            log_func("  • [Notice] Optimizer state NOT found in checkpoint. Starting optimizer with fresh momentum buffers.")

    # 2. AMP GradScaler state
    if scaler is not None:
        if has_scaler:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            log_func("  • AMP GradScaler state restored successfully.")
        else:
            log_func("  • [Notice] AMP GradScaler state NOT found in checkpoint. Initialized fresh GradScaler.")

    # 3. Learning rate scheduler state
    if scheduler is not None:
        if has_sched:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            log_func("  • Learning rate scheduler state restored successfully.")
        else:
            log_func("  • [Notice] Scheduler state NOT found in checkpoint. Initialized fresh scheduler.")

    # 4. Best Val BMA & Early Stopping
    if "best_val_bma" in checkpoint and checkpoint["best_val_bma"] is not None:
        best_val_bma = float(checkpoint["best_val_bma"])
        log_func(f"  • Restored best_val_bma: {best_val_bma*100:.2f}%")
    elif "val_bma" in checkpoint and checkpoint["val_bma"] is not None:
        best_val_bma = float(checkpoint["val_bma"])
        log_func(f"  • Inferred best_val_bma from checkpoint val_bma: {best_val_bma*100:.2f}%")
    else:
        best_val_bma = -1.0

    if "epochs_no_improve" in checkpoint and checkpoint["epochs_no_improve"] is not None:
        epochs_no_improve = int(checkpoint["epochs_no_improve"])
        log_func(f"  • Restored early-stopping counter (epochs_no_improve): {epochs_no_improve}")
    else:
        epochs_no_improve = 0

    ckpt_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = ckpt_epoch + 1
    is_exact_resume = (len(missing_states) == 0)

    if not is_exact_resume:
        log_func("!" * 75)
        log_func("[WARNING] EXACT-STATE RESUME UNAVAILABLE")
        log_func(f"Checkpoint '{checkpoint_path.name}' lacks: {', '.join(missing_states)}.")
        log_func("Optimizer momentum buffers, AMP scaler state, and learning rate schedule cannot be mathematically recovered.")
        log_func(f"Proceeding with a clearly labelled WEIGHTS-ONLY WARM START resuming at Epoch {start_epoch:02d}.")
        log_func("Note: If exact mathematical determinism without optimizer resets is required, a fresh training run is recommended.")
        log_func("!" * 75)
    else:
        log_func(f"[Resume] Exact training state successfully verified and restored from '{checkpoint_path.name}'. Resuming at Epoch {start_epoch:02d}.")

    return {
        "start_epoch": start_epoch,
        "checkpoint_epoch": ckpt_epoch,
        "best_val_bma": best_val_bma,
        "epochs_no_improve": epochs_no_improve,
        "is_exact_resume": is_exact_resume,
        "missing_states": missing_states,
        "raw_checkpoint": checkpoint
    }

def run_training_pipeline(
    data_dir: str,
    backbone: str = "efficientnet_b0",
    loss_name: str = "weighted_ce",
    epochs: int = 25,
    batch_size: int = 8,
    accum_steps: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    num_workers: int = 0,
    output_dir: Optional[str] = "backend/app/training/artifacts",
    seed: int = 42,
    device_name: Optional[str] = None,
    dry_run: bool = False,
    resume_from: Optional[str] = None
):
    """
    Executes the full training, validation, early stopping, test evaluation, and ONNX export.
    Supports Automatic Mixed Precision (AMP), Gradient Accumulation, and safe checkpoint resumption.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError:
        raise ImportError(
            "PyTorch and torchvision are required to execute training.\n"
            "Install for NVIDIA CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "Or install for CPU: pip install torch torchvision\n"
            "Then install ONNX tools: pip install onnx onnxruntime"
        )

    resume_path = None
    if resume_from:
        resume_path = Path(resume_from).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if output_dir is None:
            output_dir = str(resume_path.parent)
    elif output_dir is None:
        output_dir = "backend/app/training/artifacts"

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(seed)

    log_file_path = output_path / "training_log.txt"
    is_resuming = (resume_path is not None)
    log_mode = "a" if (is_resuming and log_file_path.exists()) else "w"
    log_file = open(log_file_path, log_mode, encoding="utf-8")

    def log_print(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    if log_mode == "a":
        log_print("\n" + "=" * 75)
        log_print("DERMORA AI - CLINICAL TRAINING PIPELINE (PyTorch) [RESUMING SESSION]")
        log_print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Architecture: {backbone} | Loss: {loss_name} | Target Epochs: {epochs}")
        log_print(f"Resuming from checkpoint: {resume_path}")
        log_print("=" * 75)
    else:
        log_print("=" * 75)
        log_print("DERMORA AI - CLINICAL TRAINING PIPELINE (PyTorch)")
        log_print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Architecture: {backbone} | Loss: {loss_name} | Seed: {seed}")
        log_print("=" * 75)

    temp_synthetic_dir = None
    if dry_run:
        log_print("\n[Notice] Running in DRY-RUN / SMOKE TEST mode with synthetic validation samples...")
        import tempfile
        from PIL import Image
        temp_synthetic_dir = Path(tempfile.mkdtemp(prefix="dermora_dryrun_"))
        images_dir = temp_synthetic_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        classes = list(HAM10000_LABEL_MAP.keys())
        for c_idx, c_code in enumerate(classes):
            for lesion_idx in range(3):
                lesion_id = f"SYN_LES_{c_code}_{lesion_idx}"
                for img_idx in range(2):
                    img_id = f"SYN_IMG_{c_code}_{lesion_idx}_{img_idx}"
                    color = ((c_idx * 35) % 255, (100 + c_idx * 20) % 255, (200 - c_idx * 25) % 255)
                    dummy_img = Image.new("RGB", (224, 224), color=color)
                    dummy_img.save(images_dir / f"{img_id}.jpg")
                    rows.append({
                        "lesion_id": lesion_id,
                        "image_id": img_id,
                        "dx": c_code,
                        "dx_type": "consensus",
                        "age": 45.0,
                        "sex": "unknown",
                        "localization": "back"
                    })
        meta_df = pd.DataFrame(rows)
        meta_csv = temp_synthetic_dir / "HAM10000_metadata.csv"
        meta_df.to_csv(meta_csv, index=False)
        data_path = temp_synthetic_dir
        status = {
            "is_complete": True,
            "metadata_path": str(meta_csv),
            "image_directories": [str(images_dir)],
            "total_images_found": len(rows),
            "expected_images": len(rows),
            "licence": "Synthetic Smoke Test Partition",
            "citation": "Synthetic unit test tensor partition"
        }
    else:
        # 1. Verify Genuine Dataset Files
        data_path = Path(data_dir)
        status = verify_dataset_directory(data_path)
        if not status["is_complete"]:
            log_file.close()
            error_details = "\n  - ".join(status.get("validation_errors", []))
            raise FileNotFoundError(
                f"Dataset at '{data_dir}' is incomplete or failed validation.\n"
                f"Images found: {status['total_images_found']} (expected >= 10000).\n"
                f"Validation errors:\n  - {error_details if error_details else 'Missing files or insufficient images.'}\n"
                "Please verify HAM10000_metadata (.tab/.csv) and image directories exist.\n"
                "Or pass '--dry-run' to test the full training code on synthetic verification tensors."
            )

    log_print(f"Dataset verified: {status['total_images_found']} images found across {len(status['image_directories'])} directories.")
    log_print(f"Metadata file: {status['metadata_path']} ({status.get('metadata_rows', 'N/A')} records)")
    log_print(f"Licence: {status['licence']}")
    if status.get("licence_warning"):
        log_print(f"Licence Notice: {status['licence_warning']}")
    if status.get("validation_warnings"):
        for warn in status["validation_warnings"]:
            log_print(f"Validation Warning: {warn}")

    # 2. Leakage-Free Stratified Group Splitting
    reused_splits = False
    if is_resuming:
        candidate_dirs = [output_path, resume_path.parent]
        for cdir in candidate_dirs:
            t_csv = cdir / "train_split.csv"
            v_csv = cdir / "val_split.csv"
            te_csv = cdir / "test_split.csv"
            if t_csv.is_file() and v_csv.is_file() and te_csv.is_file():
                log_print(f"\n[Stage 1/5] Reusing existing stratified data splits from '{cdir}' to preserve partition integrity...")
                train_df = pd.read_csv(t_csv)
                val_df = pd.read_csv(v_csv)
                test_df = pd.read_csv(te_csv)
                if cdir != output_path:
                    train_df.to_csv(output_path / "train_split.csv", index=False)
                    val_df.to_csv(output_path / "val_split.csv", index=False)
                    test_df.to_csv(output_path / "test_split.csv", index=False)
                reused_splits = True
                log_print(f"  • Train: {len(train_df)} images ({train_df['lesion_id'].nunique()} lesions)")
                log_print(f"  • Val:   {len(val_df)} images ({val_df['lesion_id'].nunique()} lesions)")
                log_print(f"  • Test:  {len(test_df)} images ({test_df['lesion_id'].nunique()} lesions)")
                break

    if not reused_splits:
        log_print("\n[Stage 1/5] Creating leakage-free stratified group splits (grouped by lesion_id)...")
        train_df, val_df, test_df = create_stratified_group_splits(
            metadata_path=status["metadata_path"],
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=seed
        )
        train_df.to_csv(output_path / "train_split.csv", index=False)
        val_df.to_csv(output_path / "val_split.csv", index=False)
        test_df.to_csv(output_path / "test_split.csv", index=False)
        log_print(f"  • Train: {len(train_df)} images ({train_df['lesion_id'].nunique()} lesions)")
        log_print(f"  • Val:   {len(val_df)} images ({val_df['lesion_id'].nunique()} lesions)")
        log_print(f"  • Test:  {len(test_df)} images ({test_df['lesion_id'].nunique()} lesions)")

    # 3. Class Imbalance Weighting & Transforms
    log_print("\n[Stage 2/5] Configuring data augmentations and class imbalance weights...")
    class_weights = compute_balanced_class_weights(train_df)
    log_print(f"  • Balanced class weights (smoothed): {np.round(class_weights, 3).tolist()}")
    
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_dirs = [Path(d) for d in status["image_directories"]]
    train_dataset = HAM10000TorchDataset(train_df, image_dirs, transform=train_transforms)
    val_dataset = HAM10000TorchDataset(val_df, image_dirs, transform=eval_transforms)
    test_dataset = HAM10000TorchDataset(test_df, image_dirs, transform=eval_transforms)

    # 4. Device Selection
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    is_cuda = (device.type == "cuda")
    log_print(f"\n[Stage 3/5] Target computation device: {device}")

    # Windows safety: num_workers=0 avoids spawn deadlocks; pin_memory enabled only on CUDA
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=is_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=is_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=is_cuda)

    log_print(f"Building {backbone} model...")
    model = build_classifier_model(backbone_name=backbone, num_classes=7, pretrained=(not dry_run and not is_resuming))
    model.to(device)

    criterion = build_loss_criterion(loss_name, class_weights=class_weights)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    # Automatic Mixed Precision for memory efficiency on 2GB GPUs (e.g. MX230)
    use_amp = is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    if use_amp:
        log_print("  • Automatic Mixed Precision (AMP) ENABLED (optimizing VRAM for 2GB GPU).")

    history = []
    history_path = output_path / "training_history.json"
    is_exact_resume = True

    if is_resuming:
        resume_info = inspect_and_load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_backbone=backbone,
            log_func=log_print
        )
        start_epoch = resume_info["start_epoch"]
        best_val_bma = resume_info["best_val_bma"]
        epochs_no_improve = resume_info["epochs_no_improve"]
        is_exact_resume = resume_info["is_exact_resume"]

        if history_path.is_file():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    prev_history = json.load(f)
                    if isinstance(prev_history, list):
                        history = [h for h in prev_history if h.get("epoch", 0) < start_epoch]
                        log_print(f"  • Restored {len(history)} previous epoch history records from '{history_path.name}'.")
            except Exception as hist_err:
                log_print(f"  • [Notice] Could not load prior training history: {hist_err}")
    else:
        start_epoch = 1
        best_val_bma = -1.0
        epochs_no_improve = 0

    best_checkpoint_path = output_path / "best_model.pt"
    latest_checkpoint_path = output_path / "latest_model.pt"

    # 5. Training Loop with Early Stopping
    log_print(f"\n[Stage 4/5] Starting training loop (epochs {start_epoch:02d} to {epochs:02d}, patience={patience}, accum={accum_steps})...")

    if start_epoch > epochs:
        log_print(f"[Notice] Requested epochs ({epochs}) already completed in checkpoint (epoch {start_epoch - 1}).")
        log_print("Skipping training loop and proceeding to test evaluation and model export.")
    else:
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            epoch_optimizer_stepped = False

            try:
                for step, (images, targets) in enumerate(train_loader):
                    images, targets = images.to(device), targets.to(device)
                    
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        if accum_steps > 1:
                            loss = loss / accum_steps

                    if use_amp:
                        scaler.scale(loss).backward()
                        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                            scale_before = scaler.get_scale()
                            scaler.step(optimizer)
                            scaler.update()
                            if scaler.get_scale() >= scale_before:
                                epoch_optimizer_stepped = True
                            optimizer.zero_grad()
                    else:
                        loss.backward()
                        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                            optimizer.step()
                            epoch_optimizer_stepped = True
                            optimizer.zero_grad()

                    running_loss += (loss.item() * (accum_steps if accum_steps > 1 else 1.0)) * images.size(0)
            except torch.cuda.OutOfMemoryError as oom_err:
                log_print(f"\n[CUDA OOM] Out of GPU memory during epoch {epoch}!")
                log_print("Recommendation: reduce --batch-size to 4 or run with '--device cpu'.")
                log_file.close()
                raise oom_err

            if epoch_optimizer_stepped:
                scheduler.step()
            train_loss = running_loss / max(len(train_dataset), 1)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Validation phase
            model.eval()
            val_targets, val_preds = [], []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(images)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    val_preds.append(probs)
                    val_targets.extend(targets.numpy())

            val_preds_arr = np.concatenate(val_preds, axis=0)
            val_targets_arr = np.array(val_targets)
            val_eval = evaluate_predictions(val_targets_arr, val_preds_arr, split_name="Validation Split")

            epoch_bma = val_eval["balanced_accuracy"]
            epoch_f1 = val_eval["macro_f1_score"]
            epoch_time = time.time() - t0

            log_print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val BMA: {epoch_bma*100:.2f}% | Val Macro F1: {epoch_f1:.4f} | Time: {epoch_time:.1f}s"
            )

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_balanced_accuracy": epoch_bma,
                "val_macro_f1": epoch_f1,
                "val_accuracy": val_eval["overall_accuracy"]
            })

            # Check if validation metric improved
            if epoch_bma > best_val_bma:
                best_val_bma = epoch_bma
                epochs_no_improve = 0
                is_new_best = True
            else:
                epochs_no_improve += 1
                is_new_best = False

            # Checkpoint payload with all training states
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_val_bma": best_val_bma,
                "epochs_no_improve": epochs_no_improve,
                "val_bma": epoch_bma,
                "val_macro_f1": epoch_f1,
                "backbone": backbone,
                "label_map": HAM10000_LABEL_MAP,
                "class_weights": class_weights.tolist() if hasattr(class_weights, "tolist") else class_weights,
                "seed": seed
            }
            save_checkpoint_atomic(checkpoint_payload, latest_checkpoint_path)

            if is_new_best:
                save_checkpoint_atomic(checkpoint_payload, best_checkpoint_path)
                log_print(f"  --> Checkpoint saved (New Best Val BMA: {best_val_bma*100:.2f}%)")
            elif epochs_no_improve >= patience:
                log_print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    # Save training history
    with open(output_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 6. Final Evaluation on Independent Test Set
    log_print("\n[Stage 5/5] Evaluating best checkpoint on unseen Independent Test Set...")
    eval_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else (resume_path if is_resuming else latest_checkpoint_path)
    checkpoint = torch.load(eval_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_targets, test_preds = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            test_preds.append(probs)
            test_targets.extend(targets.numpy())

    test_preds_arr = np.concatenate(test_preds, axis=0)
    test_targets_arr = np.array(test_targets)
    test_report = evaluate_predictions(test_targets_arr, test_preds_arr, split_name="Independent Test Set")

    with open(output_path / "test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    log_print("=" * 75)
    log_print("FINAL INDEPENDENT TEST RESULTS:")
    log_print(f"  • Overall Accuracy:    {test_report['overall_accuracy']*100:.2f}%")
    log_print(f"  • Balanced Accuracy:   {test_report['balanced_accuracy']*100:.2f}%")
    log_print(f"  • Macro F1-Score:      {test_report['macro_f1_score']:.4f}")
    log_print(f"  • Macro Sensitivity:   {test_report['macro_sensitivity']:.4f}")
    log_print(f"  • Macro Specificity:   {test_report['macro_specificity']:.4f}")
    log_print(f"  • One-vs-Rest ROC-AUC: {test_report['roc_auc_ovr']}")
    log_print("=" * 75)

    # 7. Export to ONNX & Verify Parity
    log_print("\nExporting model to ONNX format...")
    onnx_file = output_path / "dermora_model.onnx"
    try:
        export_model_to_onnx(model.cpu(), str(onnx_file))
        parity_result = verify_onnx_parity(model.cpu(), str(onnx_file))
        log_print(f"ONNX Parity Verification: {parity_result.get('status')}")
    except Exception as export_err:
        parity_result = {"verified": False, "error": str(export_err)}
        log_print(f"ONNX Export/Parity Note: {export_err}")

    with open(output_path / "onnx_parity_check.json", "w", encoding="utf-8") as f:
        json.dump(parity_result, f, indent=2)

    summary = {
        "project": "Dermora AI",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": backbone,
        "loss_function": loss_name,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "learning_rate": lr,
        "epochs_completed": len(history),
        "best_val_bma": round(best_val_bma, 4),
        "device": str(device),
        "dry_run": dry_run,
        "resumed_from": str(resume_path) if is_resuming else None,
        "is_exact_resume": is_exact_resume if is_resuming else True,
        "best_checkpoint": str(best_checkpoint_path if best_checkpoint_path.exists() else eval_checkpoint_path),
        "onnx_model": str(onnx_file),
        "test_results": {
            "overall_accuracy": test_report["overall_accuracy"],
            "balanced_accuracy": test_report["balanced_accuracy"],
            "macro_f1": test_report["macro_f1_score"],
            "macro_sensitivity": test_report["macro_sensitivity"],
            "macro_specificity": test_report["macro_specificity"],
            "roc_auc_ovr": test_report["roc_auc_ovr"]
        }
    }
    with open(output_path / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_file.close()

    if temp_synthetic_dir and temp_synthetic_dir.exists():
        import shutil
        shutil.rmtree(temp_synthetic_dir, ignore_errors=True)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dermora AI - HAM10000 PyTorch Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/ham10000", help="Directory containing HAM10000 dataset")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet34"])
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["weighted_ce", "focal"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size (8 recommended for 2GB VRAM)")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay coefficient")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (0 for Windows)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save/load artifacts (defaults to checkpoint directory when resuming, or backend/app/training/artifacts)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run 1-epoch smoke test on synthetic tensors without dataset download")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to safely resume training from (resumes at epoch + 1)")

    args = parser.parse_args()
    chosen_output_dir = args.output_dir
    if chosen_output_dir is None:
        if args.resume:
            chosen_output_dir = str(Path(args.resume).parent)
        else:
            chosen_output_dir = "backend/app/training/artifacts"

    run_training_pipeline(
        data_dir=args.data_dir,
        backbone=args.backbone,
        loss_name=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        output_dir=chosen_output_dir,
        seed=args.seed,
        device_name=args.device,
        dry_run=args.dry_run,
        resume_from=args.resume
    )
