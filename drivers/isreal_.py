import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_isreal(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Convert input to torch tensor
    input_tensor = torch.tensor(input["input"], dtype=torch.complex64)

    # Apply torch.isreal
    result = torch.isreal(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"isreal": result.numpy()}

def tensorflow_isreal(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Specify device: CPU or GPU
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to tensorflow tensor
        input_tensor = tf.constant(input["input"], dtype=tf.complex64)

        # Check if the imaginary part is 0
        result = tf.math.equal(tf.math.imag(input_tensor), 0.0)

    return {"isreal": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1, 1 + 1j, 2 + 0j], dtype=np.complex64)
    }

    # Torch example
    torch_result = torch_isreal(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_isreal(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy array comparison
    if np.array_equal(torch_result["isreal"], tf_result["isreal"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()