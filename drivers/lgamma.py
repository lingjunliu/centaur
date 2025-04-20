import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.lgamma
    result = torch.lgamma(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"lgamma_result": result.numpy()}

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

        # Apply to TensorFlow equivalent
        result = tf.math.lgamma(input_tensor)

        return {"lgamma_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, 1.0, 1.5], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert for equal result
    if np.allclose(torch_result["lgamma_result"], tf_result["lgamma_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()