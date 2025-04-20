import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply PyTorch function torch.isnan
    result = torch.isnan(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"isnan_result": result.numpy()}

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

        # Apply TensorFlow equivalent function tf.math.is_nan
        result = tf.math.is_nan(input_tensor)

        return {"isnan_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1, float('nan'), 2], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert to compare results
    if np.array_equal(torch_result["isnan_result"], tf_result["isnan_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()