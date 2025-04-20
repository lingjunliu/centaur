import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed


def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    keepdim = input.get("keepdim", False)

    # Apply torch.amax
    result = torch.amax(input_tensor, dim, keepdim=keepdim)

    if cpu:
        result = result.cpu()

    return {"amax_result": result.numpy()}


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
        dim = input["dim"]
        keepdim = input.get("keepdim", False)

        # Apply TensorFlow equivalent
        result = tf.reduce_max(input_tensor, axis=dim, keepdims=keepdim)

    return {"amax_result": result.numpy()}


def main():
    # Example input
    input_data = {
        "input": np.random.randn(4, 4).astype(np.float32),
        "dim": 1,
        "keepdim": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    assert np.allclose(torch_result["amax_result"], tf_result["amax_result"]), "Results are not equal"
    print("equal")


if __name__ == "__main__":
    main()