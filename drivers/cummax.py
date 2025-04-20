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

    # Apply to torch.cummax
    values, indices = torch.cummax(input_tensor, dim)

    if cpu and input_tensor.is_cuda:
        values = values.cpu()
        indices = indices.cpu()

    return {
        "cummax_values": values.numpy(),
        "cummax_indices": indices.numpy()
    }


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

        # Apply TensorFlow equivalent (manual implementation of cummax)
        if dim == 0:
            cummax_values = tf.TensorArray(tf.float32, size=input_tensor.shape[0])
            cummax_indices = tf.TensorArray(tf.int32, size=input_tensor.shape[0])
            max_value = -np.inf
            for i in range(input_tensor.shape[0]):
                if input_tensor[i] > max_value:
                    max_value = input_tensor[i]
                    max_index = i
                cummax_values = cummax_values.write(i, max_value)
                cummax_indices = cummax_indices.write(i, max_index)
            cummax_values = cummax_values.stack()
            cummax_indices = cummax_indices.stack()
        else:  # dim == 1
            cummax_values = tf.TensorArray(tf.float32, size=input_tensor.shape[1])
            cummax_indices = tf.TensorArray(tf.int32, size=input_tensor.shape[1])
            for j in range(input_tensor.shape[1]):
                max_value = -np.inf
                for i in range(input_tensor.shape[0]):
                    if input_tensor[i, j] > max_value:
                        max_value = input_tensor[i, j]
                        max_index = i
                    cummax_values = cummax_values.write(i * input_tensor.shape[1] + j, max_value)
                    cummax_indices = cummax_indices.write(i * input_tensor.shape[1] + j, max_index)
            cummax_values = tf.reshape(cummax_values.stack(), input_tensor.shape)
            cummax_indices = tf.reshape(cummax_indices.stack(), input_tensor.shape)

        return {
            "cummax_values": cummax_values.numpy(),
            "cummax_indices": cummax_indices.numpy()
        }


def main():
    # Example input
    input_data = {
        "input": np.array([-0.3449, -1.5447, 0.0685, -1.5104, -1.1706, 0.2259, 1.4696, -1.3284, 1.9946, -0.8209], dtype=np.float32),
        "dim": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to compare results
    assert np.array_equal(torch_result["cummax_values"], tf_result["cummax_values"])
    assert np.array_equal(torch_result["cummax_indices"], tf_result["cummax_indices"])
    print("Results are equal")


if __name__ == "__main__":
    main()