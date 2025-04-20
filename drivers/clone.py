import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed


def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply torch.clone
    cloned_tensor = torch.clone(input_tensor)

    if not cpu:
        cloned_tensor = cloned_tensor.cpu()

    return {"cloned_tensor": cloned_tensor.numpy()}

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

        # Apply TensorFlow equivalent of torch.clone
        cloned_tensor = tf.identity(input_tensor)

        return {"cloned_tensor": cloned_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality and print result
    if np.array_equal(torch_result["cloned_tensor"], tf_result["cloned_tensor"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()