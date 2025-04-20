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

    # Apply torch.logaddexp2
    result_tensor = torch.logaddexp2(input_tensor, other_tensor)

    if cpu:
        result_tensor = result_tensor.cpu()

    return {"logaddexp2_result": result_tensor.numpy()}

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

        # Apply TensorFlow equivalent of torch.logaddexp2
        result_tensor = tf.math.log(tf.pow(2.0, input_tensor) + tf.pow(2.0, other_tensor)) / tf.math.log(2.0)

        return {"logaddexp2_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32),
        "other": np.array([0.7, -1.5, 1.0, 0.5], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert results to a common format
    torch_result_np = torch_result["logaddexp2_result"]
    tf_result_np = tf_result["logaddexp2_result"]

    # Assert and print results
    if np.allclose(torch_result_np, tf_result_np):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()