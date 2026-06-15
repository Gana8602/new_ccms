import os
import cv2
import glob
import argparse

def validate_dataset(dataset_path):
    print(f"Validating dataset at: {dataset_path}")
    
    dirs = {
        "train_img": os.path.join(dataset_path, "images", "train"),
        "val_img": os.path.join(dataset_path, "images", "val"),
        "train_lbl": os.path.join(dataset_path, "labels", "train"),
        "val_lbl": os.path.join(dataset_path, "labels", "val")
    }

    errors = []
    stats = {"images": 0, "labels": 0, "total_heads": 0, "train": 0, "val": 0}

    for split in ["train", "val"]:
        img_dir = dirs[f"{split}_img"]
        lbl_dir = dirs[f"{split}_lbl"]

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"Warning: Directories for {split} split are missing.")
            continue

        images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        stats[split] = len(images)
        stats["images"] += len(images)

        for img_path in images:
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            lbl_path = os.path.join(lbl_dir, f"{basename}.txt")

            # Check image integrity
            img = cv2.imread(img_path)
            if img is None:
                errors.append(f"Corrupt Image: {filename}")
                continue

            # Check label existence
            if not os.path.exists(lbl_path):
                errors.append(f"Missing Label: {filename} has no txt file.")
                continue
            
            stats["labels"] += 1
            
            # Parse label file
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                if not lines:
                    errors.append(f"Empty Label: {basename}.txt")
                    continue
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        errors.append(f"Invalid BBox Format: {basename}.txt line {i+1}")
                        continue
                    
                    class_id = int(parts[0])
                    if class_id != 0:
                        errors.append(f"Invalid Class ID: {basename}.txt has class {class_id} (expected 0)")
                    
                    stats["total_heads"] += 1

    print("\n" + "="*40)
    print("Dataset Summary")
    print("="*40)
    print(f"Total Images: {stats['images']}")
    print(f"Total Labels: {stats['labels']}")
    print(f"Train Split:  {stats['train']} images")
    print(f"Val Split:    {stats['val']} images")
    
    if stats['images'] > 0:
        avg_heads = stats['total_heads'] / stats['images']
        print(f"Avg heads/img:{avg_heads:.2f}")
    
    print("\n" + "="*40)
    if errors:
        print(f"FOUND {len(errors)} ERRORS:")
        for e in errors[:20]: # show max 20
            print(" -", e)
        if len(errors) > 20:
            print(f"   ... and {len(errors)-20} more.")
    else:
        print("PASS: Dataset is valid. No errors found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure and labels.")
    parser.add_argument("--path", type=str, default="../datasets/ccms_heads", help="Path to dataset root")
    args = parser.parse_args()
    
    validate_dataset(args.path)
