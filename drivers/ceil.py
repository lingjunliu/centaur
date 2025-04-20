import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]).to('cpu' if cpu else 'cuda')

    # Apply torch.ceil
    ceil_tensor = torch.ceil(input_tensor)

    if not cpu:
        ceil_tensor = ceil_tensor.cpu()

    return {"ceil_result": ceil_tensor.numpy()}

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

        # Apply TensorFlow equivalent (tf.math.ceil)
        ceil_tensor = tf.math.ceil(input_tensor)

        return {"ceil_result": ceil_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-0.6341, -1.4208, -1.0900, 0.5826], [0.9999, 2.3, -3.7, -4.1]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion check
    torch_result_array = torch_result["ceil_result"]
    tf_result_array = tf_result["ceil_result"]

    if np.array_equal(torch_result_array, tf_result_array):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()