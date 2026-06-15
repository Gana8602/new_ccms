import torch
import torchvision

def merge_detections_nms(detections, iou_threshold=0.45):
    """
    Merges a list of detections using Non-Maximum Suppression.
    
    Input detection format:
    {
        "bbox": [x1, y1, x2, y2],
        "conf": float,
        ... other fields are preserved
    }
    """
    if not detections:
        return []
        
    boxes = torch.tensor([d["bbox"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["conf"] for d in detections], dtype=torch.float32)
    
    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    keep_indices = keep_indices.numpy().tolist()
    
    merged = [detections[i] for i in keep_indices]
    return merged
