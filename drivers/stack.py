import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_stack_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Unpack input dictionary
    tensors = [torch.tensor(t).to(device) for t in input["tensors"]]
    dim = input.get("dim", 0)

    # Apply torch.stack
    stacked_tensor = torch.stack(tensors, dim=dim)

    if not cpu:
        stacked_tensor = stacked_tensor.cpu()

    return {"stacked_tensor": stacked_tensor.numpy()}

def tensorflow_stack_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = [tf.constant(t) for t in input["tensors"]]
        dim = input.get("dim", 0)

        # Apply TensorFlow equivalent
        stacked_tensor = tf.stack(tensors, axis=dim)

        return {"stacked_tensor": stacked_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([[0.5, 0.3], [0.8, 0.2]], dtype=np.float32),
            np.array([[0.6, 0.4], [0.7, 0.1]], dtype=np.float32)
        ],
        "dim": 1
    }

    # Torch example
    torch_result = torch_stack_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_stack_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.array_equal(torch_result["stacked_tensor"], tf_result["stacked_tensor"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()