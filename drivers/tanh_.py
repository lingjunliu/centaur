import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply torch.tanh
    output_tensor = torch.tanh(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"tanh_output": output_tensor.numpy()}

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

        # Apply TensorFlow equivalent tf.math.tanh
        output_tensor = tf.math.tanh(input_tensor)

        return {"tanh_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, -0.3, 0.8, -1.1], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["tanh_output"], tf_result["tanh_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()