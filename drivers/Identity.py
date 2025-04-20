import numpy as np
import random
import os

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply to torch.nn.Identity
    identity = torch.nn.Identity()
    output_tensor = identity(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return output_tensor.numpy()

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # TensorFlow equivalent of torch.nn.Identity
        output_tensor = tf.identity(input_tensor)

        return output_tensor.numpy()

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    if np.array_equal(torch_result, tf_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()