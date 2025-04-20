import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Compute the asinh
    result = torch.asinh(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"asinh": result.numpy()}

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

        # Compute the asinh
        result = tf.asinh(input_tensor)

        return {"asinh": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.1606, -1.4267, -1.0899, -1.0250], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print 'equal' or 'not equal'
    torch_asinh = torch_result["asinh"]
    tf_asinh = tf_result["asinh"]

    if np.allclose(torch_asinh, tf_asinh, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()