import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Apply torch.hypot
    hypotenuse = torch.hypot(input_tensor, other_tensor)

    if not cpu:
        hypotenuse = hypotenuse.cpu()

    return {"hypot_result": hypotenuse.numpy()}

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
        other_tensor = tf.constant(input["other"])

        # Apply TensorFlow equivalent of torch.hypot
        hypotenuse = tf.math.sqrt(tf.math.square(input_tensor) + tf.math.square(other_tensor))

        return {"hypot_result": hypotenuse.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([4.0, 3.0, 5.0], dtype=np.float32),
        "other": np.array([3.0, 4.0, 6.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print result
    torch_hypot = torch_result["hypot_result"]
    tf_hypot = tf_result["hypot_result"]

    if np.allclose(torch_hypot, tf_hypot):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()