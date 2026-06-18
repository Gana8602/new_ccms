import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

def extract_frame(video_path, frame_number=0):
    """Extract a frame from the video and return it as a PIL Image."""
    print(f"Extracting frame {frame_number} from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_number} from video")
        
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb), frame

def parse_boxes(output_text):
    """
    Parse bounding boxes from the model output.
    Looks for: <box> x1, y1, x2, y2 </box> or <box>x1, y1, x2, y2</box>
    """
    box_pattern = r"<box>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</box>"
    matches = re.findall(box_pattern, output_text)
    
    results = []
    for match in matches:
        try:
            coords = [float(c) for c in match]
            results.append({
                "label": "detected",
                "box": coords  # [x1, y1, x2, y2]
            })
        except ValueError:
            continue
            
    return results

def draw_detections(frame, detections, color=(0, 255, 0), label_prefix=""):
    """Draw bounding boxes on the OpenCV frame."""
    h, w = frame.shape[:2]
    annotated = frame.copy()
    
    for det in detections:
        box = det["box"]
        # Convert 0-1000 normalized coordinates to pixel coordinates
        x1 = int(box[0] * w / 1000.0)
        y1 = int(box[1] * h / 1000.0)
        x2 = int(box[2] * w / 1000.0)
        y2 = int(box[3] * h / 1000.0)
        
        label = label_prefix if label_prefix else det["label"]
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text = f"{label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    return annotated

def main():
    video_path = "/home/sirisha/Ganapathi/crowd1.mp4"
    temp_img_path = "/home/sirisha/Ganapathi/new_ccms/media/temp_test.jpg"
    
    # Ensure media directory exists
    os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
    
    # 1. Extract frame
    try:
        pil_img, cv2_frame = extract_frame(video_path, frame_number=0)
        pil_img.save(temp_img_path)
        print(f"Saved test frame to {temp_img_path}")
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return

    # 2. Setup LocateAnything-3B directly with AutoModel and AutoProcessor
    model_id = "/home/sirisha/Ganapathi/nvidia/LocateAnything-3B"
    print(f"Loading processor and model {model_id}...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Model loaded successfully in 4-bit on GPU!")
    except Exception as e:
        print(f"GPU/Quantization load failed: {e}. Attempting CPU load...")
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Model loaded successfully on CPU!")

    # 3. Define the different queries/possible ways to detect human and related elements
    queries = {
        "irumudi": "Locate all irumudi bags carried by people.",
        "head": "Locate all human heads.",
        "body": "Locate all human bodies.",
        "only_body": "Locate all human bodies excluding the head.",
        "combined": "Locate all irumudi, head, body, and only body."
    }
    
    # Generate BGR colors for drawing
    colors = {
        "irumudi": (0, 165, 255),  # Orange
        "head": (0, 0, 255),      # Red
        "body": (255, 0, 0),      # Blue
        "only_body": (0, 255, 0),  # Green
        "combined": (255, 0, 255)  # Purple
    }

    # Save individual outputs
    annotated_frames = {}
    
    print("\n--- Running Inference ---")
    for key, prompt in queries.items():
        print(f"\nQuerying: '{prompt}'")
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Preprocess inputs
            inputs = processor(
                text=[text],
                images=[pil_img],
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to correct device
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                    tokenizer=tokenizer
                )
                
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"Raw Output: {output_text}")
            
            detections = parse_boxes(output_text)
            count = len(detections)
            print(f"Parsed {count} detections for '{key}'")
            
            # Draw detections
            annotated = draw_detections(cv2_frame, detections, color=colors[key], label_prefix=key)
            annotated_frames[key] = {
                "frame": annotated,
                "count": count,
                "prompt": prompt,
                "raw": output_text
            }
            
            # Save annotated image
            out_path = f"/home/sirisha/Ganapathi/new_ccms/media/detected_{key}.jpg"
            cv2.imwrite(out_path, annotated)
            print(f"Saved annotated image to {out_path}")
            
        except Exception as e:
            print(f"Error running inference for {key}: {e}")

    # Print summary of counts
    print("\n--- Summary of Detections & Counts ---")
    for key, info in annotated_frames.items():
        print(f"Target: {key:<10} | Count: {info['count']:<4} | Prompt: {info['prompt']}")

if __name__ == "__main__":
    main()
