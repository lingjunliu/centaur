import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Compute sign using PyTorch
    sign_tensor = torch.sign(input_tensor)

    if not cpu:
        sign_tensor = sign_tensor.cpu()

    return { "sign": sign_tensor.numpy() }

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Compute sign using TensorFlow
        sign_tensor = tf.math.sign(input_tensor)

        return { "sign": sign_tensor.numpy() }

def main():
    # Example input
    input_data = {
        "input": np.array([0.7, -1.2, 0.0, 2.3], dtype=np.float32)
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