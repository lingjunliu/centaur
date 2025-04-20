import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

# PyTorch function
def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    chunks = input["chunks"]
    dim = input["dim"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.chunk
    chunked_tensors = torch.chunk(input_tensor, chunks, dim=dim)

    if not cpu:
        chunked_tensors = [chunk.cpu() for chunk in chunked_tensors]

    return {"chunked_tensors": [tensor.numpy() for tensor in chunked_tensors]}

# TensorFlow function
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
        chunks = input["chunks"]
        dim = input["dim"]
        
        # The shape of the input tensor
        shape = input_tensor.shape
        
        # Compute the splits along the given dimension
        size_splits = [(shape[dim] + chunks - 1) // chunks] * (chunks - 1)
        size_splits.append(shape[dim] - sum(size_splits))

        chunked_tensors = tf.split(input_tensor, size_splits, axis=dim)

        return {"chunked_tensors": [chunk.numpy() for chunk in chunked_tensors]}

def main():
    # Example input
    input_data = {
        "input": np.arange(12).reshape(3, 4),  # Creating a 3x4 tensor
        "chunks": 3,
        "dim": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_chunks = [np.array(chunk) for chunk in torch_result["chunked_tensors"]]
    tf_chunks = [np.array(chunk) for chunk in tf_result["chunked_tensors"]]

    all_equal = all(np.array_equal(t1, t2) for t1, t2 in zip(torch_chunks, tf_chunks))

    if all_equal:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()