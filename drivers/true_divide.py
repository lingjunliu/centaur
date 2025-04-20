import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_true_divide_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    dividend_tensor = torch.tensor(input["dividend"])
    divisor_tensor = torch.tensor(input["divisor"])

    # Apply torch.true_divide
    result = torch.true_divide(dividend_tensor, divisor_tensor)

    if not cpu:
        result = result.cpu()

    return {"true_divide_result": result.numpy()}

def tensorflow_true_divide_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        dividend_tensor = tf.constant(input["dividend"])
        divisor_tensor = tf.constant(input["divisor"])

        # Apply TensorFlow equivalent (tf.math.divide or tf.divide)
        result = tf.math.divide(dividend_tensor, divisor_tensor)

    return {"true_divide_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "dividend": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "divisor": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_true_divide_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_true_divide_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check if results are equal (allclose is used for float comparisons to account for potential small differences)
    if np.allclose(torch_result["true_divide_result"], tf_result["true_divide_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()