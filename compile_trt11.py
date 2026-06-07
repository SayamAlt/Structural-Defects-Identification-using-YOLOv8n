import os
import tensorrt as trt

def compile_tensorrt11_fp16():
    # Define paths based on your environment output
    onnx_path = "/content/structural_defects_runs/content/runs/detect/structural_defects/yolov8n_base/weights/best.onnx"
    engine_path = "/content/structural_defects_runs/content/runs/detect/structural_defects/yolov8n_base/weights/best.engine"
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Export to ONNX first!")

    print("--- Initializing Native TensorRT 11 Compilation Pipeline ---")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # In TensorRT 11, networks must explicitly utilize the STRONGLY_TYPED flag
    # This automatically enforces FP16/mixed-precision optimizations safely.
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    
    # Parse the ONNX graph into our network architecture
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Establish builder configurations 
    config = builder.create_builder_config()
    
    # Build and serialize the engine
    print("Building TensorRT Engine (this may take a few minutes on Tesla T4)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Engine serialization failed.")
        return

    # Save the compiled edge weights
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"🎉 Success! TensorRT 11 Edge Engine compiled at: {engine_path}")

if __name__ == "__main__":
    compile_tensorrt11_fp16()
