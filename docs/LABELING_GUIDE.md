# Dense Crowd Head Detection - Labeling Guide

## Core Objective
The goal of this dataset is **NOT** standard face detection or full-body person detection. Our goal is to detect **EVERY VISIBLE HEAD** in highly dense environments such as stadiums, festivals, and temples.

## Strict Rules

### 1. Label ONLY the Head
- **DO NOT** draw the bounding box around the shoulders or torso.
- **DO NOT** draw a full-body bounding box.
- The box should tightly encapsulate the cranium/hair and the chin.

### 2. What Counts as a Head?
You must label every possible variation of a head:
- **Front Face**
- **Side Profile**
- **Back of the Head** (EXTREMELY IMPORTANT)
- **Masked Heads** (Covid masks, scarves)
- **Heads wearing hats or helmets**
- **Tiny distant heads** (Even if it is only 10x10 pixels, if you can visually determine it's a head, label it).
- **Partially occluded heads** (If a person's head is half-blocked by a pillar or another person, label the visible portion).

### 3. Exclusions
- Do not label reflections of heads in mirrors.
- Do not label printed faces on posters or shirts.
- If a blob in the background is indistinguishable from a pole/rock/noise, leave it unlabelled.

## Tool Recommendations
To efficiently annotate dense crowds (often 300+ heads per image), we recommend:
1. **CVAT (Computer Vision Annotation Tool)**: Excellent shortcut support and AI-assisted tracking.
2. **Roboflow**: Great for team collaboration and direct dataset export into YOLOv8 format.
3. **Label Studio**: Open source and highly customizable.

**Export Format**: Ensure your final export is in `YOLO v8` PyTorch txt format.
