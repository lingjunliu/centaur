import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    vec2_tensor = torch.tensor(input["vec2"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        vec2_tensor = vec2_tensor.cuda()
    
    # Apply torch.outer function
    result_tensor = torch.outer(input_tensor, vec2_tensor)
    
    if not cpu:
        result_tensor = result_tensor.cpu()
    
    return {"outer_product": result_tensor.numpy()}

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
        vec2_tensor = tf.constant(input["vec2"])
        
        # Apply TensorFlow equivalent function
        result_tensor = tf.tensordot(input_tensor, vec2_tensor, axes=0)
        
        return {"outer_product": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "vec2": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    
    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Compare results
    torch_result_np = torch_result["outer_product"]
    tf_result_np = tf_result["outer_product"]
    
    if np.allclose(torch_result_np, tf_result_np):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()