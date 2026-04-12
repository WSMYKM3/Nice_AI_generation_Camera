# Nice AI Generation Camera (MagicCamera)

Turn a single photo into a structured scene description: **instance segmentation** (YOLOv8-seg) plus **monocular depth** (Intel DPT) → a `blocks_*.json` file you can load in **Unity** to spawn objects.

---

## What it does

| Step | What runs | Purpose |
|------|-----------|---------|
| 1 | **YOLOv8-seg** (`yolov8m-seg.pt`) | Detect COCO-class objects, masks, and boxes |
| 2 | **DPT-Large** (`Intel/dpt-large` via Hugging Face) | Depth map aligned to the image |
| 3 | **Geometry** | For each detection: normalised XY (Unity-style Y flip), depth from median inside mask, scale from box vs image size |
| 4 | **Export** | JSON: `blocks_<image_stem>.json` |

Optional `--show` saves `output_<stem>.png` (detections + depth) and opens a matplotlib window.

---

## Requirements

- **Python** 3.10+ recommended  
- **Disk / RAM**: first run downloads YOLO weights and the DPT model (several GB total possible with caches)  
- **GPU**: optional but much faster for DPT and YOLO (CUDA-enabled PyTorch if available)  
- **Internet**: first run needs Hugging Face + Ultralytics downloads  

---

## Setup

```bash
cd Nice_AI_generation_Camera

# virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Hugging Face**: `transformers` will fetch `Intel/dpt-large` on first depth pass. If you use a gated model or need auth, log in with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start) as needed.

**Large images**: edges longer than **1024 px** are resized for processing. A temporary file is written to **`/tmp/_magic_cam_resized.png`** (macOS/Linux). On Windows you may need to change that path in `main.py` or run under WSL.

---

## Usage

```bash
python main.py path/to/image.jpg
```

### CLI options

| Argument | Default | Description |
|----------|---------|-------------|
| `image` | (required) | Input image path |
| `--output` | `blocks_<stem>.json` | Output JSON path |
| `--conf` | `0.5` | YOLO confidence threshold (0–1) |
| `--show` | off | Save `output_<stem>.png` and show matplotlib figure |

### Examples

```bash
# Default output: blocks_photo.json
python main.py photo.jpg

# Custom JSON path + stricter detections + visualisation
python main.py photo.jpg --output my_scene.json --conf 0.65 --show
```

If **no objects** are detected, the script writes `{"blocks": []}` and exits successfully.

---

## Output JSON (`blocks_*.json`)

Each block matches what `Unity/BlockSpawner.cs` expects:

```json
{
  "blocks": [
    {
      "id": 0,
      "type": "person",
      "position": [-0.01, -0.40, 0.09],
      "rotation": [0, 0, 0],
      "scale": [0.06, 0.27, 0.1],
      "depth": 0.09,
      "area": 5254
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `type` | YOLO class name (COCO vocabulary) |
| `position` | `[x, y, z]` — x/y normalised roughly in **[-1, 1]** (Y flipped for Unity); z from **normalised depth** in **[0, 1]** (not metres) |
| `rotation` | Placeholder `[0, 0, 0]` (degrees) |
| `scale` | Relative size from bbox vs image; `z` is a fixed placeholder **0.1** |
| `depth` | Same depth scalar as `position[2]` |
| `area` | Mask pixel count |

---

## Unity

1. Copy your generated JSON into **`Assets/StreamingAssets/`** (e.g. `blocks.json`), or set the filename on the component.  
2. Add **`BlockSpawner`** to a GameObject.  
3. Map `type` strings to prefabs in the **Prefab Map**; unmapped types use **defaultPrefab** or a generated cube.  
4. Tune **worldScale** and **scaleMultiplier** so spawned objects sit at a sensible size and spread in your scene.

See `Unity/BlockSpawner.cs` for the full API (`SpawnAll`, etc.).

---

## Project layout

| Path | Role |
|------|------|
| `main.py` | CLI pipeline |
| `requirements.txt` | Python dependencies |
| `Unity/BlockSpawner.cs` | Unity JSON loader / spawner |
| `GOAL.md` | Longer-term product notes (some items describe an older/planned SAM+BLIP stack) |

---

## License

Add a `LICENSE` file in the repo if you publish on GitHub; model weights (YOLO, DPT) have their own upstream licenses.
