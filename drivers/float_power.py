import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_float_power_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    exponent_tensor = torch.tensor(input["exponent"])

    # Apply torch.float_power
    result = torch.float_power(input_tensor, exponent_tensor)

    if not cpu:
        result = result.cpu()

    return {"float_power_result": result.numpy()}

def tensorflow_pow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        exponent_tensor = tf.constant(input["exponent"])

        # Apply TensorFlow equivalent
        result = tf.math.pow(input_tensor, exponent_tensor)

        return {"float_power_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.2, 3.4], [2.5, 4.1]], dtype=np.float32),
        "exponent": np.array([[2.0, 1.5], [3.0, 0.5]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_float_power_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_pow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print result
    torch_result_np = torch_result["float_power_result"]
    tf_result_np = tf_result["float_power_result"]

    if np.allclose(torch_result_np, tf_result_np, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()