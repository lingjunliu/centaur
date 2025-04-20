import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"]) if isinstance(input["other"], (list, np.ndarray)) else input["other"]

    if not cpu and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        if isinstance(other_tensor, torch.Tensor):
            other_tensor = other_tensor.cuda()

    # Apply torch.ge
    result = torch.ge(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

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
        other_tensor = tf.constant(input["other"]) if isinstance(input["other"], (list, np.ndarray)) else input["other"]

        # Apply TensorFlow equivalent (tf.greater_equal)
        result = tf.math.greater_equal(input_tensor, other_tensor)

        return {"result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": [[1, 2], [3, 4]],
        "other": [[1, 1], [4, 4]],
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare results
    assert np.array_equal(torch_result["result"], tf_result["result"]), "Results are not equal"
    if np.array_equal(torch_result["result"], tf_result["result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()