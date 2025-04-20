import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    keepdim = input.get("keepdim", False)

    # Compute minimum values along specified dimension(s)
    min_values = torch.amin(input_tensor, dim=dim, keepdim=keepdim)
    
    if not cpu:
        min_values = min_values.cpu()

    return {"amin": min_values.numpy()}

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
        dim = input["dim"]
        keepdim = input.get("keepdim", False)

        # Compute minimum values along specified dimension(s)
        min_values = tf.reduce_min(input_tensor, axis=dim, keepdims=keepdim)

        return {"amin": min_values.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.6451, -0.4866, 0.2987, -1.3312], 
                           [-0.5744, 1.2980, 1.8397, -0.2713], 
                           [0.9128, 0.9214, -1.7268, -0.2995], 
                           [0.9023, 0.4853, 0.9075, -1.6165]], dtype=np.float32),
        "dim": 1,
        "keepdim": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_min = torch_result["amin"]
    tf_min = tf_result["amin"]
    
    if np.allclose(torch_min, tf_min):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()