import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    padding = input["padding"]

    # Move tensor to device if not CPU
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply ReplicationPad3d
    padder = torch.nn.ReplicationPad3d(padding)
    output_tensor = padder(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()
    
    output_numpy = output_tensor.numpy()

    return {"replication_pad3d_output": output_numpy}

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
        padding = input["padding"]

        # Convert padding to a format compatible with tf.pad
        if isinstance(padding, int):
            padding = [(padding, padding)] * 3
        elif isinstance(padding, (tuple, list)) and len(padding) == 6:
            padding = [(padding[i * 2], padding[i * 2 + 1]) for i in range(3)]
        else:
            raise ValueError("Padding should be either an int or a tuple of length 6.")
        
        # Apply replication padding using tf.pad
        padding = [(0, 0), (0, 0)] + padding  # Account for batch and channel dimensions
        output_tensor = tf.pad(input_tensor, padding, mode='SYMMETRIC')
        
        output_numpy = output_tensor.numpy()

    return {"replication_pad3d_output": output_numpy}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 1, 4, 4, 4),  # Shape: (N, C, D, H, W)
        "padding": (1, 1, 1, 1, 1, 1)
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Compare results
    torch_output = torch_result["replication_pad3d_output"]
    tf_output = tf_result["replication_pad3d_output"]
    
    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()