import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def process_single_video(model_path: str, video_source: str, base_output_dir: str = "/content/predictions"):
    """
    Processes an input video track, calculates real-time performance metrics,
    and isolates all outputs into a uniquely named subfolder.
    """
    # Parse the input video name to create a clean isolated subfolder
    video_filename = os.path.basename(video_source)                 
    video_name, _ = os.path.splitext(video_filename)          
    
    # Create target folder paths
    target_folder = os.path.join(base_output_dir, video_name)
    os.makedirs(target_folder, exist_ok=True)
    
    output_video_path = os.path.join(target_folder, f"{video_name}_annotated.mp4")
    metrics_log_path = os.path.join(target_folder, f"{video_name}_metrics.csv")

    # Initialize Hardware Models and Video I/O streams
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open input video track: {video_source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"\n--- Processing Started: {video_filename} ---")
    print(f"📁 Output Target Directory: {target_folder}")
    
    frame_count = 0
    performance_records = []

    # Execution Pipeline Loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Run inference using compiled TensorRT edge weights
        results = model.predict(frame, verbose=False, device=0)
        result = results[0]
        
        # Extract individual layer pipeline latencies (stored in milliseconds)
        pre_proc = result.speed.get('preprocess', 0.0)
        infer_time = result.speed.get('inference', 0.0)
        post_proc = result.speed.get('postprocess', 0.0)
        
        # Compute pure execution frequency rates
        inference_fps = 1000.0 / infer_time if infer_time > 0 else 0.0
        
        # Cache records for validation profiling
        performance_records.append({
            "frame": frame_count,
            "preprocess_ms": pre_proc,
            "inference_ms": infer_time,
            "postprocess_nms_ms": post_proc,
            "pure_inference_fps": inference_fps
        })
        
        if frame_count % 60 == 0:
            print(f" 🟩 Frame {frame_count} | Pure Inference Speed: {inference_fps:.1f} FPS")
            
        # Draw bounding boxes and text strings onto the frame canvas array
        annotated_frame = result.plot(labels=True, boxes=True)
        
        # Frame Text Layout Configuration for HUD
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0) 
        thickness = 2
        
        cv2.putText(annotated_frame, f"Pure Inference FPS: {inference_fps:.1f}", (40, 50), font, font_scale, color, thickness)
        cv2.putText(annotated_frame, f"Pre-processing Latency: {pre_proc:.2f} ms", (40, 90), font, font_scale, color, thickness)
        cv2.putText(annotated_frame, f"Post-processing (NMS): {post_proc:.2f} ms", (40, 130), font, font_scale, color, thickness)
        
        # Stream frame chunk straight down onto storage space
        out.write(annotated_frame)
        
    # Resource Cleanup and Logging File Exports
    cap.release()
    out.release()
    
    # Save the analytical benchmark report as a structured CSV
    df = pd.DataFrame(performance_records)
    df.to_csv(metrics_log_path, index=False)
    
    print(f"Complete! isolated files generated cleanly.")
    print(f"Video: {output_video_path}")
    print(f"Metrics Summary: {metrics_log_path}")
    print(f"Average Engine Processing FPS: {df['pure_inference_fps'].mean():.2f}\n")


if __name__ == "__main__":
    MODEL_ENGINE = "/content/structural_defects_runs/content/runs/detect/structural_defects/yolov8n_base/weights/best.engine"
    INPUT_VIDEOS = [
        "/content/YTDown_Shorts_hotpot-construction-asphaltpaving-aesthe_Media_bw9nnjQWIkY_001_1080p.mp4",
        "/content/YTDown_YouTube_FastPatch-Asphalt-Alligatoring-Crack-Rep_Media_tWl2gorRJVE_001_1080p.mp4"
    ]
    
    # Iteratively evaluate each target file track sequentially
    for video_path in INPUT_VIDEOS:
        if os.path.exists(video_path):
            process_single_video(MODEL_ENGINE, video_source=video_path, base_output_dir="/content/predictions")
        else:
            print(f"⚠️ Source file mismatch, skipping path: {video_path}")