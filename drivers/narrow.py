import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    start = input["start"]
    length = input["length"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.narrow
    narrowed_tensor = torch.narrow(input_tensor, dim, start, length)
    
    if not cpu:
        narrowed_tensor = narrowed_tensor.cpu()
        
    return {"narrowed_tensor": narrowed_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Device logic
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input["dim"]
        start = input["start"]
        length = input["length"]
        
        # Apply the slicing logic
        slices = [slice(None)] * len(input_tensor.shape)
        slices[dim] = slice(start, start + length)
        narrowed_tensor = input_tensor[tuple(slices)]

        return {"narrowed_tensor": narrowed_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "dim": 1,
        "start": 1,
        "length": 2
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results and print 'equal' or 'not equal'
    assert np.array_equal(torch_result["narrowed_tensor"], tf_result["narrowed_tensor"]), "Results not equal"
    print("equal")

if __name__ == "__main__":
    main()