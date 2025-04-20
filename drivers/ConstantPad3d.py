import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

# Helper function to convert padding format
def convert_padding(padding):
    if isinstance(padding, int):
        # Converting single integer padding into a tuple of 6 elements
        return (padding, padding, padding, padding, padding, padding)
    return padding

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    padding = convert_padding(input["padding"])
    value = input["value"]

    # Apply to torch.nn.ConstantPad3d
    pad = torch.nn.ConstantPad3d(padding, value)
    result_tensor = pad(input_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"ConstantPad3d_result": result_tensor.detach().numpy()}

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
        padding = convert_padding(input["padding"])
        value = input["value"]

        # Convert PyTorch padding order to TensorFlow padding order
        tf_padding = [
            [0, 0],  # Batch dimension
            [0, 0],  # Channel dimension
            [padding[4], padding[5]],  # Depth dimension
            [padding[2], padding[3]],  # Height dimension
            [padding[0], padding[1]]   # Width dimension
        ]

        # Apply to TensorFlow equivalent
        result_tensor = tf.pad(input_tensor, tf_padding, constant_values=value)

        return {"ConstantPad3d_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(16, 3, 10, 20, 30).astype(np.float32),
        "padding": (3, 3, 6, 6, 0, 1),
        "value": 3.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["ConstantPad3d_result"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["ConstantPad3d_result"])

    # Compare the results
    assert np.allclose(torch_result["ConstantPad3d_result"], tf_result["ConstantPad3d_result"], atol=1e-6), "The results are not equal."

    print("equal")

if __name__ == "__main__":
    main()