import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    tensor_input = torch.tensor(input["input"])
    dim = input["dim"]
    
    if not cpu:
        tensor_input = tensor_input.cuda()
    
    result = torch.logcumsumexp(tensor_input, dim=dim)
    
    if not cpu:
        result = result.cpu()
    
    return { 'output': result.numpy() }

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        tensor_input = tf.constant(input["input"])
        dim = input["dim"]
        # Compute cumulative sum of the exponentials
        cumsum = tf.math.cumsum(tf.math.exp(tensor_input), axis=dim)
        # Take the log of the result
        logcumsumexp = tf.math.log(cumsum)
        
        return { 'output': logcumsumexp.numpy() }

def main():
    # Example input
    input_data = {
        "input": np.random.randn(10, 5).astype(np.float32),  # generate some random data
        "dim": 1  # dimension to perform logcumsumexp along
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Assert the results are equal
    if np.allclose(torch_result["output"], tf_result["output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()