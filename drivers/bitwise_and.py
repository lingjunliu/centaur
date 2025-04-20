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

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Apply torch.bitwise_and
    result = torch.bitwise_and(input_tensor, other_tensor)
    
    if not cpu:
        result = result.cpu()

    return {"bitwise_and_result": result.numpy()}

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
        
        # TensorFlow does not have a direct bitwise_and function, using logical_and with integer casting as an alternative
        result = tf.bitwise.bitwise_and(input_tensor, other_tensor)

        return {"bitwise_and_result": result.numpy()}


def main():
    # Example input
    input_data = {
        "input": np.array([1, 5, 7], dtype=np.int32),
        "other": np.array([1, 2, 3], dtype=np.int32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result["bitwise_and_result"], tf_result["bitwise_and_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()