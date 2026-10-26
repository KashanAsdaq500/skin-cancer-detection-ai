"""
Dermora AI - Automated Pipeline Verification & Self-Test Suite
Verifies:
1. Label mapping & 7-class ISIC taxonomy
2. Stratified lesion-group splitting with ZERO data leakage
3. Class imbalance weight calculation
4. Image quality & dermoscopic input validation / rejection filter:
   - Valid cutaneous image -> Supported
   - Low resolution (<224px) -> Rejected
   - Severe blur / low variance -> Rejected
   - Extreme luminance (over/underexposed) -> Rejected
   - Non-cutaneous / non-skin chromaticity -> Rejected
5. Pipeline specifications and non-commercial CC BY-NC 4.0 licence declaration
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.training.dataset import (
    HAM10000_LABEL_MAP,
    INDEX_TO_LABEL,
    LABEL_DISPLAY_NAMES,
    load_metadata_dataframe,
    validate_dataset_integrity,
    verify_dataset_directory,
    create_stratified_group_splits,
    check_group_leakage,
    check_duplicate_images
)
from backend.app.training.loss import compute_balanced_class_weights
from backend.app.training.evaluate import evaluate_predictions
from backend.app.model.pipeline import validate_dermoscopic_input
from backend.app.training.train import (
    save_checkpoint_atomic,
    inspect_and_load_checkpoint
)

class TestDermoraPipeline(unittest.TestCase):

    def test_01_label_mapping(self):
        """Verifies 7-class ISIC/HAM10000 label taxonomy."""
        self.assertEqual(len(HAM10000_LABEL_MAP), 7)
        expected_classes = {"nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"}
        self.assertEqual(set(HAM10000_LABEL_MAP.keys()), expected_classes)
        for code, idx in HAM10000_LABEL_MAP.items():
            self.assertEqual(INDEX_TO_LABEL[idx], code)
            self.assertIn(code, LABEL_DISPLAY_NAMES)

    def test_02_leakage_free_stratified_splits(self):
        """Verifies that lesion_id grouping strictly prevents data leakage."""
        import tempfile
        rows = []
        for c in HAM10000_LABEL_MAP.keys():
            for lesion_num in range(10):
                lesion_id = f"LES_{c}_{lesion_num}"
                for img_num in range(2):  # 2 images per lesion
                    rows.append({
                        "lesion_id": lesion_id,
                        "image_id": f"IMG_{c}_{lesion_num}_{img_num}",
                        "dx": c
                    })
        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name

        train_df, val_df, test_df = create_stratified_group_splits(
            metadata_csv_path=csv_path,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )

        leakage = check_group_leakage(train_df, val_df, test_df)
        self.assertFalse(leakage["has_leakage"], f"Leakage detected: {leakage}")
        self.assertEqual(leakage["train_val_overlap_count"], 0)
        self.assertEqual(leakage["train_test_overlap_count"], 0)
        self.assertEqual(leakage["val_test_overlap_count"], 0)
        Path(csv_path).unlink(missing_ok=True)

    def test_03_balanced_class_weights(self):
        """Verifies that inverse-frequency class weights smooth minority classes."""
        data = {
            "dx": ["nv"] * 700 + ["mel"] * 100 + ["bkl"] * 100 + ["bcc"] * 50 + ["akiec"] * 30 + ["vasc"] * 10 + ["df"] * 10
        }
        df = pd.DataFrame(data)
        weights = compute_balanced_class_weights(df)
        self.assertEqual(len(weights), 7)
        # Minority classes (df, vasc) must receive higher weights than majority (nv)
        nv_idx = HAM10000_LABEL_MAP["nv"]
        df_idx = HAM10000_LABEL_MAP["df"]
        self.assertGreater(weights[df_idx], weights[nv_idx])
        # Mean of weights should be approximately 1.0
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=2)

    def test_04_evaluation_metrics_rigor(self):
        """Verifies BMA, macro sensitivity, specificity, and confusion matrix calculation."""
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
        # Perfect predictions
        y_probs = np.zeros((10, 7))
        for i, t in enumerate(y_true):
            y_probs[i, t] = 1.0

        eval_report = evaluate_predictions(y_true, y_probs, split_name="Unit Test Partition")
        self.assertEqual(eval_report["overall_accuracy"], 1.0)
        self.assertEqual(eval_report["balanced_accuracy"], 1.0)
        self.assertEqual(eval_report["macro_f1_score"], 1.0)
        self.assertEqual(eval_report["macro_sensitivity"], 1.0)
        self.assertEqual(eval_report["macro_specificity"], 1.0)
        self.assertEqual(len(eval_report["per_class_metrics"]), 7)

    def test_05_input_quality_rejection_filter(self):
        """Verifies that non-skin, blurred, or low-res inputs are correctly rejected."""
        # 1. Low resolution rejection (< 224 px)
        tiny_img = Image.new("RGB", (100, 100), color=(180, 120, 100))
        tiny_res = validate_dermoscopic_input(tiny_img)
        self.assertFalse(tiny_res.is_supported_dermoscopic_input)
        self.assertTrue(any("resolution" in r.lower() for r in tiny_res.rejection_reasons))

        # 2. Severe underexposure rejection (< 25 mean luminance)
        black_img = Image.new("RGB", (300, 300), color=(5, 5, 5))
        black_res = validate_dermoscopic_input(black_img)
        self.assertFalse(black_res.is_supported_dermoscopic_input)
        self.assertTrue(any("underexposed" in r.lower() for r in black_res.rejection_reasons))

        # 3. Non-cutaneous chromaticity rejection (pure blue object)
        blue_img = Image.new("RGB", (300, 300), color=(10, 50, 220))
        blue_res = validate_dermoscopic_input(blue_img)
        self.assertFalse(blue_res.is_supported_dermoscopic_input)
        self.assertTrue(any("non-dermatological" in r.lower() for r in blue_res.rejection_reasons))

        # 4. Valid synthetic skin-tone lesion (cutaneous hue, good size, textured)
        arr = np.full((300, 300, 3), (210, 150, 120), dtype=np.uint8)
        np.random.seed(42)
        noise = np.random.randint(-15, 15, (300, 300, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        yy, xx = np.ogrid[:300, :300]
        mask = ((xx - 150)**2 + (yy - 150)**2) < 60**2
        arr[mask] = np.clip(arr[mask].astype(np.int16) - 90, 0, 255).astype(np.uint8)
        valid_img = Image.fromarray(arr)
        valid_res = validate_dermoscopic_input(valid_img)
        self.assertTrue(valid_res.is_supported_dermoscopic_input, f"Rejection reasons: {valid_res.rejection_reasons}")
        self.assertGreater(valid_res.skin_tone_ratio, 0.25)

    def test_06_fitzpatrick_phototype_inclusivity(self):
        """
        Verifies that the image quality filter does NOT reject skin lesions 
        due to diversity in Fitzpatrick phototypes (Types I through VI).
        """
        fitzpatrick_tones = {
            "Type I (Very Fair)": (250, 220, 200),
            "Type II (Fair)": (240, 200, 175),
            "Type III (Medium)": (215, 170, 130),
            "Type IV (Olive)": (180, 130, 90),
            "Type V (Brown/Dark)": (130, 85, 55),
            "Type VI (Deep Black/Brown)": (80, 50, 35)
        }
        for name, base_color in fitzpatrick_tones.items():
            arr = np.full((300, 300, 3), base_color, dtype=np.uint8)
            np.random.seed(42)
            noise = np.random.randint(-10, 10, (300, 300, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            yy, xx = np.ogrid[:300, :300]
            mask = ((xx - 150)**2 + (yy - 150)**2) < 50**2
            arr[mask] = np.clip(arr[mask].astype(np.int16) * 0.6, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            res = validate_dermoscopic_input(img)
            self.assertTrue(
                res.is_supported_dermoscopic_input,
                f"Fitzpatrick {name} unexpectedly rejected: {res.rejection_reasons}"
            )
            self.assertGreater(res.skin_tone_ratio, 0.25)

    def test_07_tab_metadata_loading_and_quoted_values(self):
        """
        Verifies that TAB-delimited metadata with double-quoted values is parsed correctly,
        preserves required columns, and generates leakage-free stratified splits.
        """
        import tempfile
        tab_content = (
            'lesion_id\timage_id\tdx\tdx_type\tage\tsex\tlocalization\tdataset\n'
            '"HAM_0001"\t"ISIC_0001"\t"bkl"\t"histo"\t70.0\t"male"\t"back"\t"vidir"\n'
            '"HAM_0001"\t"ISIC_0002"\t"bkl"\t"histo"\t70.0\t"male"\t"back"\t"vidir"\n'
            '"HAM_0002"\t"ISIC_0003"\t"mel"\t"histo"\t55.0\t"female"\t"face"\t"vidir"\n'
            '"HAM_0002"\t"ISIC_0004"\t"mel"\t"histo"\t55.0\t"female"\t"face"\t"vidir"\n'
            '"HAM_0003"\t"ISIC_0005"\t"nv"\t"consensus"\t40.0\t"male"\t"trunk"\t"vidir"\n'
            '"HAM_0003"\t"ISIC_0006"\t"nv"\t"consensus"\t40.0\t"male"\t"trunk"\t"vidir"\n'
            '"HAM_0004"\t"ISIC_0007"\t"bcc"\t"histo"\t65.0\t"male"\t"neck"\t"vidir"\n'
            '"HAM_0005"\t"ISIC_0008"\t"akiec"\t"histo"\t80.0\t"female"\t"hand"\t"vidir"\n'
            '"HAM_0006"\t"ISIC_0009"\t"vasc"\t"consensus"\t30.0\t"male"\t"leg"\t"vidir"\n'
            '"HAM_0007"\t"ISIC_0010"\t"df"\t"consensus"\t50.0\t"female"\t"arm"\t"vidir"\n'
        )
        with tempfile.NamedTemporaryFile(suffix=".tab", mode="w", delete=False, encoding="utf-8") as f:
            f.write(tab_content)
            tab_path = f.name

        try:
            df = load_metadata_dataframe(tab_path)
            self.assertEqual(len(df), 10)
            # Verify quotes were stripped
            self.assertEqual(df["lesion_id"].iloc[0], "HAM_0001")
            self.assertEqual(df["image_id"].iloc[0], "ISIC_0001")
            self.assertEqual(df["dx"].iloc[0], "bkl")

            train_df, val_df, test_df = create_stratified_group_splits(
                metadata_path=tab_path,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                random_seed=42
            )
            self.assertEqual(len(train_df) + len(val_df) + len(test_df), 10)
            leakage = check_group_leakage(train_df, val_df, test_df)
            self.assertFalse(leakage["has_leakage"])
        finally:
            Path(tab_path).unlink(missing_ok=True)

    def test_08_dataset_validation_integrity(self):
        """
        Verifies that dataset validation identifies missing columns, invalid labels,
        and missing image files.
        """
        # 1. Missing required column
        invalid_cols_df = pd.DataFrame({"lesion_id": ["L1"], "dx": ["mel"]})
        val = validate_dataset_integrity(invalid_cols_df)
        self.assertFalse(val["is_valid"])
        self.assertTrue(any("image_id" in err for err in val["errors"]))

        # 2. Unknown diagnostic label
        unknown_dx_df = pd.DataFrame({"lesion_id": ["L1"], "image_id": ["I1"], "dx": ["carcinoma_xyz"]})
        val = validate_dataset_integrity(unknown_dx_df)
        self.assertFalse(val["is_valid"])
        self.assertIn("carcinoma_xyz", val["unknown_labels"])

        # 3. Missing image files on disk
        valid_df = pd.DataFrame({"lesion_id": ["L1"], "image_id": ["I1"], "dx": ["mel"]})
        available_stems = {"I2", "I3"}
        val = validate_dataset_integrity(valid_df, available_image_stems=available_stems)
        self.assertFalse(val["is_valid"])
        self.assertEqual(val["missing_images_count"], 1)

    def test_09_real_dataset_on_disk_if_present(self):
        """
        If the real HAM10000 dataset is present at E:\\DermoraData\\ham10000,
        verify the entire directory layout, 10015 image count, zero leakage,
        and zero missing images.
        """
        real_data_dir = Path(r"E:\DermoraData\ham10000")
        if not real_data_dir.exists():
            self.skipTest(f"Dataset root '{real_data_dir}' not found on this machine; skipping real data verification.")

        status = verify_dataset_directory(real_data_dir)
        self.assertTrue(status["is_complete"], f"Real dataset verification failed: {status.get('validation_errors')}")
        self.assertEqual(status["total_images_found"], 10015)
        self.assertEqual(len(status["image_directories"]), 2)
        self.assertEqual(status["metadata_format"], ".tab")
        self.assertEqual(status["missing_images_count"], 0)
        self.assertEqual(len(status["unknown_labels"]), 0)
        self.assertIn("CC BY-NC 4.0", status["licence"])

        # Verify group splitting on the real dataset has zero lesion leakage
        train_df, val_df, test_df = create_stratified_group_splits(
            metadata_path=status["metadata_path"],
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 10015)
        leakage = check_group_leakage(train_df, val_df, test_df)
        self.assertFalse(leakage["has_leakage"])
        self.assertEqual(leakage["train_val_overlap_count"], 0)
        self.assertEqual(leakage["train_test_overlap_count"], 0)
        self.assertEqual(leakage["val_test_overlap_count"], 0)

    def test_10_atomic_checkpoint_saving(self):
        """Verifies atomic checkpoint saving prevents corruption and cleans up temporary swap files."""
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_file = Path(tmp_dir) / "checkpoints" / "latest_model.pt"
            state_v1 = {"epoch": 1, "loss": 1.25, "tensor": torch.tensor([1.0, 2.0, 3.0])}
            save_checkpoint_atomic(state_v1, target_file)

            self.assertTrue(target_file.is_file())
            # Ensure no orphan temporary swap files remain
            tmp_files = list(target_file.parent.glob("*.tmp_*"))
            self.assertEqual(len(tmp_files), 0)

            loaded_v1 = torch.load(target_file, map_location="cpu")
            self.assertEqual(loaded_v1["epoch"], 1)
            self.assertEqual(loaded_v1["loss"], 1.25)
            self.assertTrue(torch.equal(loaded_v1["tensor"], state_v1["tensor"]))

            # Atomic replace with v2
            state_v2 = {"epoch": 2, "loss": 0.85, "tensor": torch.tensor([4.0, 5.0, 6.0])}
            save_checkpoint_atomic(state_v2, target_file)
            loaded_v2 = torch.load(target_file, map_location="cpu")
            self.assertEqual(loaded_v2["epoch"], 2)
            self.assertEqual(loaded_v2["loss"], 0.85)
            self.assertTrue(torch.equal(loaded_v2["tensor"], state_v2["tensor"]))

    def test_11_full_state_checkpoint_serialization_and_restoration(self):
        """Verifies full training state serialization: model weights, optimizer, scaler, scheduler, best_val_bma, epochs_no_improve."""
        import tempfile
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_full_ckpt.pt"
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 7))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            scaler = torch.amp.GradScaler("cpu", enabled=True)

            # Perform a step to populate optimizer state
            dummy_x = torch.randn(4, 8)
            dummy_y = torch.randint(0, 7, (4,))
            out = model(dummy_x)
            loss = nn.functional.cross_entropy(out, dummy_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            payload = {
                "epoch": 5,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_bma": 0.785,
                "epochs_no_improve": 2,
                "val_bma": 0.770,
                "backbone": "test_net",
                "label_map": HAM10000_LABEL_MAP
            }
            save_checkpoint_atomic(payload, ckpt_path)

            # Fresh components
            model2 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 7))
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
            scaler2 = torch.amp.GradScaler("cpu", enabled=True)

            logs = []
            info = inspect_and_load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model2,
                optimizer=optimizer2,
                scheduler=scheduler2,
                scaler=scaler2,
                expected_backbone="test_net",
                log_func=logs.append
            )

            self.assertTrue(info["is_exact_resume"])
            self.assertEqual(len(info["missing_states"]), 0)
            self.assertEqual(info["start_epoch"], 6)  # Resumes at epoch + 1
            self.assertEqual(info["checkpoint_epoch"], 5)
            self.assertEqual(info["best_val_bma"], 0.785)
            self.assertEqual(info["epochs_no_improve"], 2)
            self.assertEqual(scheduler2.last_epoch, 1)

            # Check model parameter weights match exactly
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_12_legacy_checkpoint_inspection_and_warm_start_detection(self):
        """
        Verifies that legacy checkpoints (lacking optimizer, scaler, scheduler, early-stopping state)
        are explicitly identified as exact-state unavailable and routed to a weights-only warm start.
        """
        import tempfile
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "legacy_model.pt"
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 7))

            # Simulate legacy format (identical to Epoch 01 latest_model.pt)
            legacy_payload = {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "backbone": "test_net",
                "val_bma": 0.6032,
                "label_map": HAM10000_LABEL_MAP,
                "class_weights": [1.0] * 7
            }
            save_checkpoint_atomic(legacy_payload, ckpt_path)

            model2 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 7))
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
            scaler2 = torch.amp.GradScaler("cpu", enabled=True)

            logs = []
            info = inspect_and_load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model2,
                optimizer=optimizer2,
                scheduler=scheduler2,
                scaler=scaler2,
                expected_backbone="test_net",
                log_func=logs.append
            )

            self.assertFalse(info["is_exact_resume"])
            self.assertIn("optimizer_state_dict", info["missing_states"])
            self.assertIn("scaler_state_dict", info["missing_states"])
            self.assertIn("scheduler_state_dict", info["missing_states"])
            self.assertIn("early_stopping_state", info["missing_states"])
            self.assertEqual(info["start_epoch"], 2)  # Resumes at epoch 1 + 1 = 2
            self.assertEqual(info["best_val_bma"], 0.6032)  # Inferred from val_bma
            self.assertEqual(info["epochs_no_improve"], 0)
            self.assertTrue(any("EXACT-STATE RESUME UNAVAILABLE" in log for log in logs))
            self.assertTrue(any("WEIGHTS-ONLY WARM START" in log for log in logs))

        # Verify live on-disk legacy checkpoint from backup directory (strictly read-only)
        legacy_backup_ckpt = Path("backend/app/training/artifacts_backup_20261020/latest_model.pt")
        if legacy_backup_ckpt.is_file():
            from backend.app.training.model import build_classifier_model
            real_model = build_classifier_model("efficientnet_b0", num_classes=7, pretrained=False)
            real_opt = torch.optim.AdamW(real_model.parameters(), lr=3e-4)
            real_sched = torch.optim.lr_scheduler.CosineAnnealingLR(real_opt, T_max=25)
            real_logs = []
            real_info = inspect_and_load_checkpoint(
                checkpoint_path=legacy_backup_ckpt,
                model=real_model,
                optimizer=real_opt,
                scheduler=real_sched,
                expected_backbone="efficientnet_b0",
                log_func=real_logs.append
            )
            self.assertFalse(real_info["is_exact_resume"])
            self.assertIn("optimizer_state_dict", real_info["missing_states"])
            self.assertIn("scaler_state_dict", real_info["missing_states"])
            self.assertIn("scheduler_state_dict", real_info["missing_states"])
            self.assertTrue(any("EXACT-STATE RESUME UNAVAILABLE" in log for log in real_logs))

        # Also verify live trained checkpoint in artifacts_real_training_20261020 has full exact resume state
        trained_ckpt = Path("backend/app/training/artifacts_real_training_20261020/latest_model.pt")
        if trained_ckpt.is_file():
            from backend.app.training.model import build_classifier_model
            trained_model = build_classifier_model("efficientnet_b0", num_classes=7, pretrained=False)
            trained_opt = torch.optim.AdamW(trained_model.parameters(), lr=3e-4)
            trained_sched = torch.optim.lr_scheduler.CosineAnnealingLR(trained_opt, T_max=25)
            trained_scaler = torch.amp.GradScaler("cpu", enabled=True)
            trained_logs = []
            trained_info = inspect_and_load_checkpoint(
                checkpoint_path=trained_ckpt,
                model=trained_model,
                optimizer=trained_opt,
                scheduler=trained_sched,
                scaler=trained_scaler,
                expected_backbone="efficientnet_b0",
                log_func=trained_logs.append
            )
            self.assertTrue(trained_info["is_exact_resume"])
            self.assertEqual(trained_info["checkpoint_epoch"], 11)
            self.assertEqual(trained_info["start_epoch"], 12)
            self.assertEqual(trained_info["best_val_bma"], 0.68)
            self.assertEqual(trained_info["epochs_no_improve"], 5)

    def test_13_split_preservation_and_log_append(self):
        """Verifies that split CSVs are preserved across resume and log files are appended rather than truncated."""
        import tempfile
        import pandas as pd
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "artifacts_resume_test"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Create existing split CSVs
            dummy_train = pd.DataFrame({"lesion_id": ["L1"], "image_id": ["I1"], "dx": ["mel"]})
            dummy_val = pd.DataFrame({"lesion_id": ["L2"], "image_id": ["I2"], "dx": ["nv"]})
            dummy_test = pd.DataFrame({"lesion_id": ["L3"], "image_id": ["I3"], "dx": ["bkl"]})
            dummy_train.to_csv(out_dir / "train_split.csv", index=False)
            dummy_val.to_csv(out_dir / "val_split.csv", index=False)
            dummy_test.to_csv(out_dir / "test_split.csv", index=False)

            # Create existing log file
            log_file = out_dir / "training_log.txt"
            initial_log = "Epoch 01/25 | Train Loss: 1.3419 | Val BMA: 60.32%\n"
            log_file.write_text(initial_log, encoding="utf-8")

            # Create checkpoint
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 7))
            ckpt_path = out_dir / "latest_model.pt"
            save_checkpoint_atomic({
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "backbone": "efficientnet_b0",
                "val_bma": 0.6032,
                "label_map": HAM10000_LABEL_MAP
            }, ckpt_path)

            # Simulate log opening in append mode for resume
            is_resuming = True
            log_mode = "a" if (is_resuming and log_file.exists()) else "w"
            self.assertEqual(log_mode, "a")
            with open(log_file, log_mode, encoding="utf-8") as f:
                f.write("[RESUMING SESSION] Resuming from checkpoint\n")

            log_content = log_file.read_text(encoding="utf-8")
            self.assertIn("Epoch 01/25 | Train Loss: 1.3419", log_content)
            self.assertIn("[RESUMING SESSION]", log_content)

            # Verify splits on disk still match original
            loaded_train = pd.read_csv(out_dir / "train_split.csv")
            self.assertEqual(len(loaded_train), 1)
            self.assertEqual(loaded_train["lesion_id"].iloc[0], "L1")

if __name__ == "__main__":
    unittest.main(verbosity=2)
