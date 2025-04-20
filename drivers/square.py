import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_square(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.square
    squared_tensor = torch.square(input_tensor)

    if not cpu:
        squared_tensor = squared_tensor.cpu()

    # Return as numpy array for consistent comparison
    return squared_tensor.numpy()

def tensorflow_square(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply to TensorFlow equivalent: tf.square
        squared_tensor = tf.square(input_tensor)

        return squared_tensor.numpy()

def main():
    # Example input
    input_data = {
        "input": np.array([-2.0755, 1.0226, 0.0831, 0.4806], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_square(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_square(input_data)
    print("TensorFlow result:", tf_result)

    # Assert results are equal and print appropriate message
    if np.allclose(torch_result, tf_result, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()