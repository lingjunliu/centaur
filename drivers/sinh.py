import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Ensure input is a tensor
    input_tensor = torch.tensor(input["input"])

    if not cpu and torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')

    # Apply torch.sinh
    result_tensor = torch.sinh(input_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"sinh": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Ensure input is a tensor
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent (tf.sinh)
        result_tensor = tf.math.sinh(input_tensor)

        return {"sinh": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, -0.5, 1.0, -1.0, 0.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_result_array = np.array(torch_result["sinh"])
    tf_result_array = np.array(tf_result["sinh"])

    if np.allclose(torch_result_array, tf_result_array, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()