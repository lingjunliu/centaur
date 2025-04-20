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

    # Apply the torch.nn.functional.leaky_relu function
    output_tensor = torch.nn.functional.leaky_relu(input_tensor, negative_slope=negative_slope, inplace=inplace)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"leaky_relu_output": output_tensor.numpy()}


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

        # Apply the TensorFlow equivalent operation
        output_tensor = tf.nn.leaky_relu(input_tensor, alpha=negative_slope)

        return {"leaky_relu_output": output_tensor.numpy()}


def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, -0.9]], dtype=np.float32),
        "negative_slope": 0.05,
        "inplace": False  # Note: TensorFlow does not have an equivalent 'inplace' parameter
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = np.array(torch_result["leaky_relu_output"])
    tf_output = np.array(tf_result["leaky_relu_output"])

    # Ensure outputs are the same shape and values
    assert np.allclose(torch_output, tf_output), "Outputs are not equal!"

    if np.allclose(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()