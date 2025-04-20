import torch
import tensorflow as tf
import numpy as np

# Assuming you have a set_seed function in src.setseed
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.bool)
    other_tensor = torch.tensor(input["other"], dtype=torch.bool)

    # Perform logical_or operation
    output_tensor = torch.logical_or(input_tensor, other_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"logical_or_result": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.bool)
        other_tensor = tf.constant(input["other"], dtype=tf.bool)

        # Perform logical_or operation
        output_tensor = tf.logical_or(input_tensor, other_tensor)

        return {"logical_or_result": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[True, False, True], [False, True, False]], dtype=bool),
        "other": np.array([[False, False, True], [True, False, True]], dtype=bool)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_output = torch_result["logical_or_result"]
    tf_output = tf_result["logical_or_result"]

    if np.array_equal(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()