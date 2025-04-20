import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Perform bitwise OR
    result = torch.bitwise_or(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"bitwise_or_result": result.numpy()}  # Convert to numpy array for common comparison format

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
        other_tensor = tf.constant(input["other"])

        # Perform bitwise OR
        result = tf.bitwise.bitwise_or(input_tensor, other_tensor)

        return {"bitwise_or_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "other": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check equality
    assert np.array_equal(torch_result["bitwise_or_result"], tf_result["bitwise_or_result"]), "Results do not match!"
    print("equal")

if __name__ == "__main__":
    main()