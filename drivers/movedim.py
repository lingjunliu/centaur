import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

# PyTorch implementation
def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    source = input["source"]
    destination = input["destination"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.movedim
    result = torch.movedim(input_tensor, source, destination)

    if not cpu:
        result = result.cpu()

    return {"movedim_result": result.numpy()}

# TensorFlow implementation
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
        source = input["source"]
        destination = input["destination"]

        # Apply tf.experimental.numpy.moveaxis as TensorFlow equivalent
        result = tf.experimental.numpy.moveaxis(input_tensor, source, destination)

    return {"movedim_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(3, 2, 1),
        "source": 1,
        "destination": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results, converting to numpy arrays
    if np.array_equal(torch_result["movedim_result"], tf_result["movedim_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()