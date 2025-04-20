import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dims = input["dims"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.flip
    result = torch.flip(input_tensor, dims)
    
    if not cpu:
        result = result.cpu()

    return {"flipped_tensor": result.numpy()}

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
        dims = input["dims"]

        # Apply TensorFlow equivalent
        result = tf.reverse(input_tensor, axis=dims)

        return {"flipped_tensor": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.arange(8).reshape(2, 2, 2).astype(np.float32),
        "dims": [0, 1]
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Convert results to compare
    torch_output_np = np.array(torch_result["flipped_tensor"])
    tf_output_np = np.array(tf_result["flipped_tensor"])

    assert np.allclose(torch_output_np, tf_output_np), "Torch and TensorFlow results do not match."

    print("equal")

if __name__ == "__main__":
    main()