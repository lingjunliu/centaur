import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    padding = input["padding"]
    value = input["value"]

    # Apply ConstantPad2d
    padder = torch.nn.ConstantPad2d(padding, value)
    padded_tensor = padder(input_tensor)

    if not cpu:
        padded_tensor = padded_tensor.cpu()

    return {"padded_tensor": padded_tensor.numpy()}

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
        padding = input["padding"]
        value = input["value"]

        # Apply the equivalent padding operation in TensorFlow
        if isinstance(padding, int):
            padding = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        else:
            padding = [[0, 0], [0, 0], [padding[2], padding[3]], [padding[0], padding[1]]]

        padded_tensor = tf.pad(input_tensor, paddings=padding, constant_values=value)

        return {"padded_tensor": padded_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(1, 1, 2, 2).astype(np.float32),  # Shape (N, C, H, W)
        "padding": (2, 2, 2, 2),  # (padding_left, padding_right, padding_top, padding_bottom)
        "value": 3.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["padded_tensor"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["padded_tensor"])

    np.testing.assert_allclose(torch_result["padded_tensor"], tf_result["padded_tensor"], rtol=1e-5)
    print("equal")

if __name__ == "__main__":
    main()