import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    min_val = input.get("min", None)
    max_val = input.get("max", None)
    
    # Apply to torch.clamp
    if min_val is None and max_val is None:
        clamped_tensor = input_tensor
    elif min_val is None:
        clamped_tensor = torch.clamp(input_tensor, max=max_val)
    elif max_val is None:
        clamped_tensor = torch.clamp(input_tensor, min=min_val)
    else:
        clamped_tensor = torch.clamp(input_tensor, min=min_val, max=max_val)

    if not cpu:
        clamped_tensor = clamped_tensor.cpu()

    return {"clamped_tensor": clamped_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        min_val = input.get("min", None)
        max_val = input.get("max", None)
        
        # Apply to TensorFlow equivalent
        if min_val is None and max_val is None:
            clamped_tensor = input_tensor
        elif min_val is None:
            clamped_tensor = tf.clip_by_value(input_tensor, clip_value_min=-float('inf'), clip_value_max=max_val)
        elif max_val is None:
            clamped_tensor = tf.clip_by_value(input_tensor, clip_value_min=min_val, clip_value_max=float('inf'))
        else:
            clamped_tensor = tf.clip_by_value(input_tensor, clip_value_min=min_val, clip_value_max=max_val)
        
        return {"clamped_tensor": clamped_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-1.7120, 0.1734, -0.0478, -0.0922]], dtype=np.float32),
        "min": -0.5,
        "max": 0.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_array = torch_result["clamped_tensor"]
    tf_array = tf_result["clamped_tensor"]

    # Assertion to compare the results
    assert np.allclose(torch_array, tf_array), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()