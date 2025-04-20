import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_mul(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"]) if np.isscalar(input["other"]) is False else input["other"]
    
    # Perform multiplication
    if np.isscalar(other_tensor):
        result = torch.mul(input_tensor, other_tensor)
    else:
        other_tensor = torch.tensor(input["other"])
        result = torch.mul(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"mul_result": result.numpy()}

def tensorflow_mul(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    # Unpack input dictionary
    input_tensor = tf.constant(input["input"])
    other_tensor = tf.constant(input["other"]) if np.isscalar(input["other"]) is False else input["other"]

    with tf.device(device_string):
        # Perform multiplication
        if np.isscalar(other_tensor):
            result = tf.multiply(input_tensor, other_tensor)
        else:
            other_tensor = tf.constant(input["other"])
            result = tf.multiply(input_tensor, other_tensor)

    return {"mul_result": result.numpy()}

def main():
    # Example input 1: Scalar multiplication
    input_data1 = {
        "input": np.array([0.2015, -0.4255, 2.6087], dtype=np.float32),
        "other": 100
    }

    # Example input 2: Broadcasting multiplication
    input_data2 = {
        "input": np.random.randn(4, 1).astype(np.float32),
        "other": np.random.randn(1, 4).astype(np.float32)
    }

    # Torch example 1
    torch_result1 = torch_mul(input_data1)
    print("Torch result 1:", torch_result1)

    # TensorFlow example 1
    tf_result1 = tensorflow_mul(input_data1)
    print("TensorFlow result 1:", tf_result1)

    # Torch example 2
    torch_result2 = torch_mul(input_data2)
    print("Torch result 2:", torch_result2)

    # TensorFlow example 2
    tf_result2 = tensorflow_mul(input_data2)
    print("TensorFlow result 2:", tf_result2)

    # Assert and compare results for Example 1
    result1_equality = np.allclose(torch_result1["mul_result"], tf_result1["mul_result"], atol=1e-6)
    print("Example 1 results equal:", "equal" if result1_equality else "not equal")

    # Assert and compare results for Example 2
    result2_equality = np.allclose(torch_result2["mul_result"], tf_result2["mul_result"], atol=1e-6)
    print("Example 2 results equal:", "equal" if result2_equality else "not equal")

if __name__ == "__main__":
    main()