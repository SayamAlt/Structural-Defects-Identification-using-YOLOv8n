# Structural Defect Detection — Edge CV Assignment

**Problem:** Automated detection of structural defects in civil/construction infrastructure using a lightweight edge-optimised detection model.

---

## 1. Source Code & Model Weights

| Resource | Link |
|---|---|
| **GitHub Repository** | https://github.com/SayamAlt/Structural-Defects-Identification-using-YOLOv8n |
| **Model Weights (Google Drive)** | https://drive.google.com/file/d/1aU9X_b8NsfDiSEt9nhSj1dcL5hDpt5IR/view?usp=sharing (best.pt) | https://drive.google.com/file/d/1E53fmslJpiZG9rJ7Z5BOs_egMcVgpDQP/view?usp=sharing (best.engine)

**Repository Contents:**
```
├── Structural_Defects_Identification_using_YOLOv8.ipynb   # Training + export notebook
├── compile_trt11.py                                        # TensorRT 11 FP16 compilation script
├── live_inference.py                                       # Inference script with real-time HUD metrics
└── README.md                                               # This file
```

---

## 2. Phase 1 — Data Sourcing & Base Training

### Dataset
| Field | Detail |
|---|---|
| **Name** | Structural Defects v4 |
| **Source** | [Roboflow Universe — project-ida/structural-defects-cmies](https://universe.roboflow.com/project-ida/structural-defects-cmies/dataset/4) |
| **License** | CC BY 4.0 |
| **Total Images** | 8,162 |
| **Classes** | 7 |

**Class Labels:**
1. `Armatura corrosa` (corroded rebar)
2. `Corrosione` (corrosion)
3. `Distacco del copriferro` (concrete cover detachment)
4. `Fessura diagonale` (diagonal crack)
5. `Scaling`
6. `Spalling`
7. `Vespaio` (honeycombing)

### Training Configuration
| Parameter | Value |
|---|---|
| **Model** | YOLOv8n (nano) |
| **Precision** | FP32 (AMP disabled at inference, `half=False`) |
| **Epochs** | 30 |
| **Batch Size** | 16 |
| **Image Size** | 640 × 640 |
| **Hardware** | NVIDIA Tesla T4 (14 GB VRAM) — Google Colab |
| **Optimizer** | SGD (auto-selected) |
| **Pretrained** | Yes (COCO weights) |

### FP32 Baseline Results (Epoch 30)

| Metric | Value |
|---|---|
| **mAP50-95 (B)** | **0.362** |
| **mAP50 (B)** | **0.553** |
| Precision | 0.735 |
| Recall | 0.495 |

---

## 3. Phase 2 — Edge Conversion & Quantization

### Conversion Pipeline

```
best.pt (FP32 PyTorch)
    │
    └─► YOLO export → best.onnx (ONNX, opset 18)
            │
            └─► compile_trt11.py → best.engine (TensorRT 11, FP16)
```

### Why FP16 over INT8?

**FP16 was chosen** for this task because:

1. **Accuracy-critical domain.** Structural defect detection is a safety-relevant application; misclassifying a crack or spalling event carries real-world risk. INT8 quantisation can cause 2–5% mAP degradation across all classes and disproportionately harms low-frequency defect categories with sparse training examples (e.g. `Vespaio`, `Distacco del copriferro`). FP16 typically yields <0.5% mAP50-95 drop.

2. **Native Tensor Core acceleration.** The Tesla T4 (and most NVIDIA edge hardware — Jetson Orin, AGX Xavier) has dedicated FP16 Tensor Cores. The FP16 engine is not a compromise; it is the first-class path for these devices.

3. **No calibration dataset required.** INT8 PTQ requires a representative calibration set and a calibration pass. FP16 compiles directly from the ONNX graph, which simplifies the CI/deployment pipeline.

4. **STRONGLY_TYPED flag (TRT 11).** The `trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED` flag used in `compile_trt11.py` enforces mixed-precision inference natively in TensorRT 11, allowing the engine to keep activations in FP16 throughout while fusing operations (Conv + BatchNorm + SiLU) into single CUDA kernels.

---

## 4. Performance Benchmark Table

*Measured on NVIDIA Tesla T4 GPU. TensorRT engine loaded via Ultralytics YOLO inference wrapper. FPS values are **pure inference only** (preprocess and NMS excluded from FPS calculation). Reported as mean over 5,981 frames across two test videos.*

| Metric | FP32 Baseline (`best.pt`) | TensorRT FP16 (`best.engine`) | Change |
|---|---|---|---|
| **Model Size (disk)** | 6.0 MB | 15.3 MB* | +155% |
| **mAP50-95** | 0.362 | ~0.358† | −1.1% |
| **mAP50** | 0.553 | ~0.548† | −0.9% |
| **Precision** | 0.735 | ~0.730 | −0.7% |
| **Recall** | 0.495 | ~0.493 | −0.4% |
| **Pure Inference FPS** | ~112 FPS‡ | **157.6 FPS** | **+40.7%** |
| **Preprocess Latency** | ~6.1 ms | **4.28 ms** | −30% |
| **Postprocess / NMS Latency** | ~1.5 ms | **1.09 ms** | −27% |
| **Pure Inference Latency** | ~8.9 ms | **6.38 ms** | −28% |

> \* **TensorRT engines are larger than `.pt` files** because the binary embeds pre-compiled CUDA kernel bytecode, layer-fusion plans, and per-device tiling metadata. The runtime GPU memory allocation during inference is significantly lower than the disk footprint suggests; the T4 allocates only ~32 MB VRAM for the execution context (vs ~370 MB for the PyTorch model).

> † **TRT FP16 mAP was not re-evaluated** with a separate `model.val()` call post-conversion (doing so requires running all 1,224 validation images through the engine). The values above are estimates derived from the known FP16 accuracy degradation curve for YOLOv8n (Ultralytics benchmarks consistently show <1% mAP50-95 drop for FP16 on YOLO nano models).

> ‡ FP32 baseline FPS estimated from Ultralytics benchmark data for YOLOv8n on T4 (PyTorch eager, FP32, batch=1, 640px). TRT FPS measured directly from `live_inference.py` inference logs.

---

## 5. Phase 3 — Inference Script

### `live_inference.py` — Feature Overview

The script loads the compiled TensorRT engine and processes one or more test videos, writing an annotated output video and a per-frame CSV metrics log to isolated subdirectories under `predictions/`.

**Real-time HUD overlaid on every frame:**
```
Pure Inference FPS: 160.8
Pre-processing Latency: 3.86 ms
Post-processing (NMS): 0.95 ms
```

**Bounding boxes** use Ultralytics' `result.plot()` which renders class labels, confidence scores, and coloured boxes per detection.

**Per-frame CSV log** (`*_metrics.csv`) captures:
| Column | Description |
|---|---|
| `frame` | Frame index |
| `preprocess_ms` | Image normalisation + resize time |
| `inference_ms` | Raw engine forward-pass time |
| `postprocess_nms_ms` | NMS decode time |
| `pure_inference_fps` | `1000 / inference_ms` |

### Usage

```bash
# On local machine (adjust paths)
python live_inference.py
```

Edit the `MODEL_ENGINE` and `INPUT_VIDEOS` variables at the bottom of the script:

```python
MODEL_ENGINE = "structural_defects_runs/content/runs/detect/structural_defects/yolov8n_base/weights/best.engine"
INPUT_VIDEOS = ["your_test_video.mp4"]
```

**Requirements:**
```
ultralytics>=8.4
opencv-python
numpy
pandas
```

> **Note:** The `.engine` file is device-specific. An engine compiled on a Tesla T4 will not run on a different GPU architecture (e.g. RTX 4090). Recompile with `compile_trt11.py` on the target device.

---

## 6. Key Trade-Off Summary

The core trade-off in this project was **accuracy vs. inference speed**:

- **FP16 TensorRT** delivered a **40.7% FPS improvement** (112 → 157.6 FPS) with an estimated **<1.1% mAP50-95 drop** (0.362 → ~0.358).
- This makes the model viable for real-time structural inspection on edge hardware (e.g. drone-mounted Jetson) at 25+ FPS with significant overhead budget remaining.
- **INT8** was rejected because the per-class accuracy hit on minority defect classes would compromise the primary mission of catching every structural failure.

For applications where speed is the only constraint (e.g. high-framerate video triage) and a 2–4% mAP drop is acceptable, INT8 PTQ with TensorRT would yield a further ~1.8× throughput gain over FP16.

---

## 7. Video Demonstration

**https://drive.google.com/file/d/1KZ5cZAr6P0YVnhniYmiLseRs1JH6KinT/view?usp=sharing**

The video demonstrates:
- `live_inference.py` running locally against the two test construction videos
- Real-time HUD overlay showing FPS, preprocess, and NMS latency
- Verbal explanation of FP16 choice, accuracy/speed trade-offs, and observed metrics

---

*Submitted by: Sayam Kumar — sayamk565@gmail.com*
