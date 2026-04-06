# MagicCamera — Project Goal

## Overview

Take a single photo and automatically reconstruct a scene in Unity by:

1. Segmenting every object in the image
2. Estimating the 3-D position and rotation of each object
3. Exporting a JSON file that Unity reads to spawn matching prefabs

---

## Pipeline

```
Input image
    │
    ▼
[SAM] Segment Anything Model
    │  → per-object binary masks + bounding boxes
    │
    ▼
[Depth Estimator] (e.g. MiDaS / ZoeDepth / Depth Anything)
    │  → depth map aligned to the image
    │  → sample depth inside each mask → estimated Z distance
    │
    ▼
[Caption / Classifier] (e.g. BLIP image-text-to-text)
    │  → free-form label for each segment (no fixed vocab)
    │
    ▼
[Geometry Solver]
    │  → derive world-space position  (x, y, z)
    │  → estimate rotation / normal from mask shape or depth gradient
    │
    ▼
blocks.json
    │
    ▼
[Unity C# Importer]
    └─ read JSON → spawn prefab for each block at position + rotation
```

---

## Output JSON schema (`blocks.json`)

```json
{
  "blocks": [
    {
      "id": 0,
      "type": "refrigerator",          // free-form caption label
      "position": [x, y, z],           // normalised or world-space coords
      "rotation": [rx, ry, rz],        // Euler angles in degrees (Unity convention)
      "scale":    [sx, sy, sz],        // relative size derived from mask area
      "depth":    1.45,                // estimated depth in metres (or relative units)
      "area":     52341                // mask pixel area (for debugging)
    }
  ]
}
```

---

## Current Status

| Step | Status | Notes |
|------|--------|-------|
| SAM segmentation | ✅ working | `sam_vit_b`, auto-mask, CPU |
| Free-form caption | 🔄 in progress | BLIP `image-text-to-text` pipeline |
| Depth estimation | ⬜ not started | need to pick + integrate a depth model |
| Rotation estimation | ⬜ not started | derive from depth gradient or assume upright |
| JSON export | ✅ basic version | `blocks.json`, no depth/rotation yet |
| Unity importer | ⬜ not started | C# script to read JSON and spawn prefabs |

---

## Key Decisions / Open Questions

- **Depth model**: MiDaS (lightweight, CPU-ok) vs ZoeDepth / Depth Anything v2 (more accurate, heavier)
- **Coordinate system**: decide on normalised (−1 to 1) vs metric (metres) before Unity integration
- **Rotation**: start with all objects upright (rx=rz=0, ry from mask bounding-box aspect) — improve later
- **Prefab mapping**: need a dictionary from caption label → Unity prefab name/path
- **Scale**: derive from mask bounding box relative to image size, then calibrate with depth
