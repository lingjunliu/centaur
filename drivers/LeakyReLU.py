import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    negative_slope = input.get("negative_slope", 0.01)
    inplace = input.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply LeakyReLU
    leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    if not cpu:
        leaky_relu = leaky_relu.cuda()

    result = leaky_relu(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"LeakyReLU_output": result.numpy()}

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
        negative_slope = input.get("negative_slope", 0.01)

        # Apply LeakyReLU (no direct in-place option in TensorFlow, always non-in-place)
        result = tf.nn.leaky_relu(input_tensor, alpha=negative_slope)

        return {"LeakyReLU_output": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, -0.9]], dtype=np.float32),
        "negative_slope": 0.05,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["LeakyReLU_output"], tf_result["LeakyReLU_output"], rtol=1e-5, atol=1e-8):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()