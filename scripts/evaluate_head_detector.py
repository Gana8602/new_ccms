import os
import argparse
from ultralytics import YOLO

def evaluate(model_path, imgsz, split):
    print("="*50)
    print(f"📊 Evaluating Model: {model_path}")
    print("="*50)
    
    yaml_path = os.path.abspath("../datasets/ccms_heads/head.yaml")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    
    project_dir = os.path.abspath("runs/detect")
    
    metrics = model.val(
        data=yaml_path,
        split=split,
        imgsz=imgsz,
        save_json=True,
        save_hybrid=False,
        conf=0.10,
        iou=0.45,
        project=project_dir,
        name="head_eval",
        exist_ok=True
    )
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    
    # Calculate Precision/Recall across all classes (we only have class 0)
    p = metrics.box.mp
    r = metrics.box.mr
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*40)
    
    print(f"\nPlots and Confusion Matrix saved to runs/detect/head_eval/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CCMS Head Detector")
    parser.add_argument("--model", type=str, default="/home/sirisha/Ganapathi/new_ccms/models/best_head_custom.pt", help="Path to model to evaluate")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or test)")
    
    args = parser.parse_args()
    evaluate(args.model, args.imgsz, args.split)
