import os
import argparse
from ultralytics import YOLO

def train(model_name, imgsz, epochs, batch, device):
    print("="*50)
    print("🚀 CCMS Dense Crowd Head Detector Training")
    print("="*50)
    
    yaml_path = os.path.abspath("/Volumes/tridelMac/Projects/Dev/ccms/new_ccms/datasets/ccms_heads/head.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: Dataset yaml not found at {yaml_path}")
        return

    # Initialize model
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)
    
    # Sensible augmentations for dense crowds
    augmentations = {
        "mosaic": 1.0,
        "mixup": 0.2,
        "flipud": 0.0,
        "fliplr": 0.5,
        "scale": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4
    }

    print(f"\nStarting training on {device}...")
    print(f"Epochs: {epochs} | Batch: {batch} | ImgSz: {imgsz}")
    
    # Train
    project_dir = os.path.abspath("runs/detect")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name="head_training",
        exist_ok=True,
        **augmentations
    )
    
    # Auto-copy model
    # Sometimes ultralytics alters the name slightly if exist_ok=False, but exist_ok=True keeps it exact
    best_weight = os.path.join(project_dir, "head_training", "weights", "best.pt")
    target_weight = "/Volumes/tridelMac/Projects/Dev/ccms/ccms/models/best_head_custom.pt"
    
    if os.path.exists(best_weight):
        print("\nTraining Complete! Copying best model...")
        os.system(f"cp {best_weight} {target_weight}")
        print(f"✅ Best model successfully copied to {target_weight}")
    else:
        print(f"\n❌ Training finished but best.pt not found at {best_weight}.")
        # Fallback search just in case
        print("Looking for best.pt in the runs directory...")
        os.system(f"find {project_dir} -name 'best.pt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom CCMS Head Detector")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base model (yolo11m.pt or yolo11l.pt)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (1280 or 1536 recommended)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")
    
    args = parser.parse_args()
    
    train(args.model, args.imgsz, args.epochs, args.batch, args.device)
