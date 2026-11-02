"""
train_mvtec_patchcore.py
========================
Trains PatchCore on MVTec AD dataset categories using defect-free (good) images.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

CATEGORIES = [
    "metal_nut",
    "bottle",
    "cable",
    "carpet",
    "grid",
    "leather",
    "screw",
    "tile",
    "wood",
    "zipper",
]

def train_category(category: str, root_dir: str = "mv dataset", output_dir: str = "results/mvtec_patchcore"):
    print("=" * 65)
    print(f">> Training PatchCore on MVTec Category: {category.upper()}")
    print("=" * 65)
    
    save_path = Path(output_dir) / category
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. DataModule (Loads train/good for training, test/good + defects for evaluation)
    datamodule = MVTecAD(
        root=root_dir,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
    )
    
    # 2. PatchCore Model (ResNet-18 backbone for fast CPU training & high accuracy)
    model = Patchcore(
        backbone="resnet18",
        pre_trained=True,
        coreset_sampling_ratio=0.1,
    )
    
    # 3. Engine (CPU mode)
    engine = Engine(
        default_root_dir=str(save_path),
        accelerator="cpu",
    )
    
    # 4. Train (Memory Bank Fitting on normal images only)
    print(f"\n[1/2] Fitting normal memory bank on defect-free '{category}' images...")
    engine.fit(model=model, datamodule=datamodule)
    
    # 5. Evaluate on test set (Normal vs Defects)
    print(f"\n[2/2] Evaluating anomaly detection performance on test set...")
    test_results = engine.test(model=model, datamodule=datamodule)
    
    auroc = "N/A"
    if test_results and len(test_results) > 0:
        res = test_results[0]
        auroc = res.get("image_AUROC", res.get("AUROC", "N/A"))
        print(f"\nTest Image-level AUROC for {category}: {auroc}")
        
    report = (
        f"MVTec AD PatchCore Anomaly Detection Report\n"
        f"==========================================\n"
        f"Category: {category}\n"
        f"Backbone: resnet18\n"
        f"Trained on: 100% defect-free normal images\n"
        f"Image AUROC: {auroc}\n"
    )
    report_file = save_path / "anomaly_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_file}")
    print(f"Finished training '{category}' successfully!\n")
    return auroc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument("--category", type=str, default="metal_nut", help="MVTec category to train (or 'all')")
    parser.add_argument("--data_dir", type=str, default="mv dataset", help="Path to MVTec dataset")
    args = parser.parse_args()
    
    if args.category.lower() == "all":
        print(f"Starting training for all {len(CATEGORIES)} categories sequentially...")
        for cat in CATEGORIES:
            try:
                train_category(cat, root_dir=args.data_dir)
            except Exception as e:
                print(f"Error training {cat}: {e}")
    else:
        train_category(args.category, root_dir=args.data_dir)
