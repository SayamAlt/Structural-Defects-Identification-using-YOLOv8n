import os
import sys
import cv2
import pandas as pd
import torch
from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR = os.path.join(REPO_ROOT, "structural_defects_runs",
                           "content", "runs", "detect",
                           "structural_defects", "yolov8n_base", "weights")

# Ordered preference: TensorRT engine → ONNX → PyTorch checkpoint
_CANDIDATES = [
    os.path.join(WEIGHTS_DIR, "best.engine"),
    os.path.join(WEIGHTS_DIR, "best.onnx"),
    os.path.join(WEIGHTS_DIR, "best.pt"),
]

def resolve_model() -> str:
    """Return the best available weight file for this machine."""
    # TensorRT requires Linux + CUDA; skip on other platforms
    import platform
    for path in _CANDIDATES:
        if not os.path.exists(path):
            continue
        if path.endswith(".engine"):
            if platform.system() != "Linux":
                print(f"[INFO] Skipping {os.path.basename(path)} — "
                      "TensorRT engine requires Linux + CUDA.")
                continue
            try:
                import tensorrt  # noqa: F401
            except ImportError:
                print("[INFO] Skipping best.engine — tensorrt not installed.")
                continue
        print(f"[INFO] Using model: {path}")
        return path
    raise FileNotFoundError(
        f"No usable weight file found in {WEIGHTS_DIR}. "
        "Expected best.engine / best.onnx / best.pt."
    )

def resolve_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def process_single_video(model_path: str, video_source: str,
                         base_output_dir: str) -> None:
    """
    Run inference on one video, write annotated output + per-frame CSV.
    """
    video_filename = os.path.basename(video_source)
    video_name, _ = os.path.splitext(video_filename)

    target_folder = os.path.join(base_output_dir, video_name)
    os.makedirs(target_folder, exist_ok=True)

    output_video_path = os.path.join(target_folder, f"{video_name}_annotated.mp4")
    metrics_log_path  = os.path.join(target_folder, f"{video_name}_metrics.csv")

    device = resolve_device()
    model  = YOLO(model_path)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = src_fps if src_fps > 0 else 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (width, height))

    print(f"\n--- Processing Started: {video_filename} ---")
    print(f"    Device  : {device}")
    print(f"    Model   : {os.path.basename(model_path)}")
    print(f"    Output  : {target_folder}")

    frame_count = 0
    performance_records = []

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness  = 2
    # White text with black outline for readability on any background
    text_color    = (255, 255, 255)
    outline_color = (0, 0, 0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        results = model.predict(frame, verbose=False, device=device)
        result  = results[0]

        pre_proc    = result.speed.get("preprocess",  0.0)
        infer_time  = result.speed.get("inference",   0.0)
        post_proc   = result.speed.get("postprocess", 0.0)

        inference_fps = 1000.0 / infer_time if infer_time > 0 else 0.0

        performance_records.append({
            "frame":               frame_count,
            "preprocess_ms":       pre_proc,
            "inference_ms":        infer_time,
            "postprocess_nms_ms":  post_proc,
            "pure_inference_fps":  inference_fps,
        })

        if frame_count % 60 == 0:
            print(f"  Frame {frame_count:5d} | "
                  f"Inference: {inference_fps:6.1f} FPS | "
                  f"Pre: {pre_proc:.1f} ms | NMS: {post_proc:.1f} ms")

        # Draw detections (bounding boxes + class label + confidence)
        annotated = result.plot(labels=True, boxes=True)

        # HUD lines to overlay
        hud_lines = [
            f"Pure Inference FPS : {inference_fps:.1f}",
            f"Pre-process Latency: {pre_proc:.2f} ms",
            f"Post-process (NMS) : {post_proc:.2f} ms",
        ]

        for i, text in enumerate(hud_lines):
            y = 45 + i * 40
            # Draw outline first, then white text on top
            cv2.putText(annotated, text, (20, y), font, font_scale,
                        outline_color, thickness + 2, cv2.LINE_AA)
            cv2.putText(annotated, text, (20, y), font, font_scale,
                        text_color, thickness, cv2.LINE_AA)

        out.write(annotated)

    cap.release()
    out.release()

    df = pd.DataFrame(performance_records)
    df.to_csv(metrics_log_path, index=False)

    print(f"\n[DONE] {video_filename}")
    print(f"  Annotated video : {output_video_path}")
    print(f"  Metrics CSV     : {metrics_log_path}")
    print(f"  Avg Infer FPS   : {df['pure_inference_fps'].mean():.2f}")
    print(f"  Avg Pre-proc    : {df['preprocess_ms'].mean():.2f} ms")
    print(f"  Avg NMS         : {df['postprocess_nms_ms'].mean():.2f} ms\n")

if __name__ == "__main__":
    model_path = resolve_model()
    output_dir = os.path.join(REPO_ROOT, "predictions")

    input_videos = [
        os.path.join(REPO_ROOT, "YTDown_Shorts_hotpot-construction-asphaltpaving-aesthe_Media_bw9nnjQWIkY_001_1080p.mp4"),
        os.path.join(REPO_ROOT, "YTDown_YouTube_FastPatch-Asphalt-Alligatoring-Crack-Rep_Media_tWl2gorRJVE_001_1080p.mp4"),
    ]

    found = [v for v in input_videos if os.path.exists(v)]
    if not found:
        print("[ERROR] No test video files found. Expected:")
        for v in input_videos:
            print(f"  {v}")
        sys.exit(1)

    for video_path in found:
        process_single_video(model_path, video_source=video_path,
                             base_output_dir=output_dir)