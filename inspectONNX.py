import onnx
import onnxruntime as ort
import numpy as np
import json
import torch
import os

def get_onnx_intermediate_outputs(onnx_path, dummy_input):
    """Get intermediate outputs from ONNX model"""
    # Load the ONNX model
    model = onnx.load(onnx_path)
    

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
        print("Continuing without shape inference...")
    
    # Create a session 
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Create a list of all output names to capture
    output_names = [output.name for output in model.graph.output]
    
    #  lookup dictionary from all available sources
    type_lookup = {}
    
    # Get types from value_info 
    for value_info in model.graph.value_info:
        if value_info.type.HasField('tensor_type'):
            type_lookup[value_info.name] = value_info.type.tensor_type.elem_type
    
    # Get types from initializers
    for initializer in model.graph.initializer:
        type_lookup[initializer.name] = initializer.data_type
    
    # Get types from inputs
    for graph_input in model.graph.input:
        if graph_input.type.HasField('tensor_type'):
            type_lookup[graph_input.name] = graph_input.type.tensor_type.elem_type
    
    # Get types from outputs
    for graph_output in model.graph.output:
        if graph_output.type.HasField('tensor_type'):
            type_lookup[graph_output.name] = graph_output.type.tensor_type.elem_type
    
    # Add intermediate node outputs
    for node in model.graph.node:

        skip_types = []  
        
        if node.op_type in skip_types:
            continue
        
        for output in node.output:
            if output not in output_names:
                output_names.append(output)
    
    # Create intermediate outputs 
    intermediate_outputs = []
    for name in output_names:
        # Get the tensor type from lookup, default to FLOAT if not found
        tensor_type = type_lookup.get(name, onnx.TensorProto.FLOAT)
        
        # For debugging: print if we couldn't find the type
        if name not in type_lookup:
            print(f"Warning: Could not find type for '{name}', defaulting to FLOAT")
        
        intermediate_outputs.append(onnx.helper.make_tensor_value_info(
            name, 
            tensor_type, 
            None  # Shape can be None, ONNX will infer it
        ))
    
    # Create a new model with all outputs
    graph = onnx.helper.make_graph(
        nodes=model.graph.node,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=intermediate_outputs,
        initializer=model.graph.initializer,
        value_info=model.graph.value_info
    )
    
    new_model = onnx.helper.make_model(graph)
    new_model.ir_version = model.ir_version
    
    # Clear and copy opset imports
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)
    
    new_model.producer_name = model.producer_name
    new_model.producer_version = model.producer_version
    
    # Save the temporary model
    temp_onnx_path = onnx_path.replace('.onnx', '_with_intermediates.onnx')
    
    try:
        onnx.save(new_model, temp_onnx_path)
        
        # Run the model with all outputs
        session = ort.InferenceSession(temp_onnx_path, options)
        input_name = session.get_inputs()[0].name
        
        # Convert input to numpy if it's a torch tensor
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.numpy()
        
        # Run the model
        outputs = session.run(output_names, {input_name: dummy_input})
        
        # Create a dictionary of output names to values
        output_dict = dict(zip(output_names, outputs))
        
        return output_dict
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_onnx_path):
            try:
                os.remove(temp_onnx_path)
                print(f"Cleaned up temporary file: {temp_onnx_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_onnx_path}: {e}")

def main(onnx_path, model_name="Enh_32"):
    # Create dummy input
    batch_size = 1
    in_audio_channels = 1
    name = model_name.lower()

    # decide time resolution
    if "enh" in name:
        time_samples = 16000
    elif "sep" in name:
        time_samples = 8000
    else:
        raise ValueError(f"Unknown model type in name: {model_name}")

    # decide dtype
    if "16" in name:
        dtype = torch.float16
    elif "32" in name:
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown precision in model name: {model_name}")

    # create input
    dummy_input = torch.randn(
        batch_size,
        in_audio_channels,
        time_samples,
        dtype=dtype,
    )

    print(f"Input shape: {dummy_input.shape}")
    
    # Get intermediate outputs from ONNX model
    try:
        intermediate_outputs = get_onnx_intermediate_outputs(onnx_path, dummy_input)
    except Exception as e:
        print(f"Error getting intermediate outputs: {e}")
        return
    
    # Prepare data for JSON output
    output_data = {}
    for layer_name, output_array in intermediate_outputs.items():
        # Check if array is empty
        is_empty = output_array.size == 0
        
        # Base info that always exists
        base_info = {
            "shape": list(output_array.shape),
            "dtype": str(output_array.dtype),
            "size": int(output_array.size)
        }
        
        if is_empty:
            # For empty arrays, just record basic info
            output_data[layer_name] = base_info
            print(f"{layer_name}: shape {output_array.shape}, dtype {output_array.dtype} [EMPTY]")
        elif output_array.dtype == np.bool_:
            # Handle boolean types
            output_data[layer_name] = {
                **base_info,
                "true_count": int(np.sum(output_array)),
                "false_count": int(np.sum(~output_array))
            }
            print(f"{layer_name}: shape {output_array.shape}, dtype {output_array.dtype}")
        elif np.issubdtype(output_array.dtype, np.integer):
            # Handle integer types
            output_data[layer_name] = {
                **base_info,
                "mean": float(np.mean(output_array)),
                "std": float(np.std(output_array)),
                "min": int(np.min(output_array)),
                "max": int(np.max(output_array))
            }
            print(f"{layer_name}: shape {output_array.shape}, dtype {output_array.dtype}")
        else:  # Float types
            output_data[layer_name] = {
                **base_info,
                "mean": float(np.mean(output_array)),
                "std": float(np.std(output_array)),
                "min": float(np.min(output_array)),
                "max": float(np.max(output_array))
            }
            print(f"{layer_name}: shape {output_array.shape}, dtype {output_array.dtype}")
    
    # Save to JSON file
    with open("onnx_layer_outputs_" + model_name + ".json", "w") as f:
        json.dump(output_data, f, indent=2)

    print("\nLayer outputs saved to onnx_layer_outputs_" + model_name + ".json")

    #  save sample values from each layer (first few elements)
    sample_data = {}
    for layer_name, output_array in intermediate_outputs.items():
        # Skip empty arrays
        if output_array.size == 0:
            sample_data[layer_name] = {
                "first_10_values": [],
                "note": "empty array"
            }
            continue
            
        # Flatten and take first 10 elements
        flat_array = output_array.flatten()
        
        # Handle different data types
        if output_array.dtype == np.bool_:
            sample_values = [bool(x) for x in flat_array[:10]]
        elif np.issubdtype(output_array.dtype, np.integer):
            sample_values = [int(x) for x in flat_array[:10]]
        else:
            sample_values = [float(x) for x in flat_array[:10]]
        
        sample_data[layer_name] = {
            "first_10_values": sample_values
        }

    with open("onnx_layer_samples_" + model_name + ".json", "w") as f:
        json.dump(sample_data, f, indent=2)

    print("Sample values saved to onnx_layer_samples_" + model_name + ".json")

if __name__ == "__main__":
    main("enh_32.onnx", model_name="Enh_32")