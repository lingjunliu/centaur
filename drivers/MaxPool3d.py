import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    set_seed()

    # Unpack inputs
    input_tensor = torch.tensor(input["input"])
    kernel_size = input["kernel_size"]
    stride = input.get("stride", None)
    padding = input.get("padding", 0)
    dilation = input.get("dilation", 1)
    return_indices = input.get("return_indices", False)
    ceil_mode = input.get("ceil_mode", False)

    # Apply PyTorch MaxPool3d
    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    output = pool(input_tensor)
    
    if return_indices:
        output, indices = output

    if not cpu:
        output = output.cpu()

    return {"maxpool3d_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    set_seed()

    # Set device string
    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Unpack inputs
        input_tensor = tf.constant(input["input"])
        pool_size = input["kernel_size"]
        strides = input.get("stride", 1)
        padding = 'VALID' if input.get("padding", 0) == 0 else 'SAME'
        data_format = 'channels_first'  # To match PyTorch's default NCDHW format

        # Apply TensorFlow MaxPooling3D
        pool = tf.keras.layers.MaxPooling3D(pool_size, strides=strides, padding=padding, data_format=data_format)
        output = pool(input_tensor)

    return {"maxpool3d_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 1, 4, 4, 4).astype(np.float32),  # Example 3D input
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check if the outputs are equal
    torch_output = np.array(torch_result["maxpool3d_output"])
    tf_output = np.array(tf_result["maxpool3d_output"])

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()