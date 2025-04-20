import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    equation = input["equation"]
    operands = [torch.tensor(operand) for operand in input["operands"]]

    # Apply to torch.einsum
    result = torch.einsum(equation, *operands)

    if not cpu:
        result = result.cpu()
    
    return {"einsum_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        equation = input["equation"]
        operands = [tf.constant(operand) for operand in input["operands"]]

        # Apply to TensorFlow equivalent using tf.einsum
        result = tf.einsum(equation, *operands).numpy()
      
    return {"einsum_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "equation": 'ij,jk->ik',
        "operands": [
            np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
            np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both results to a common format and compare
    torch_result_np = torch_result["einsum_result"]
    tf_result_np = tf_result["einsum_result"]

    if np.allclose(torch_result_np, tf_result_np, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()