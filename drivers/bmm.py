import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    mat2_tensor = torch.tensor(input["mat2"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        mat2_tensor = mat2_tensor.cuda()

    # Perform the batch matrix-matrix product
    result = torch.bmm(input_tensor, mat2_tensor)

    if not cpu:
        result = result.cpu()

    return {"bmm_result": result.numpy()}

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
        mat2_tensor = tf.constant(input["mat2"])

        # Perform the batch matrix-matrix product using tf.matmul
        result = tf.matmul(input_tensor, mat2_tensor)

        return {"bmm_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(10, 3, 4).astype(np.float32),
        "mat2": np.random.rand(10, 4, 5).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["bmm_result"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["bmm_result"])

    # Assertions to compare results
    torch_result_np = torch_result["bmm_result"]
    tf_result_np = tf_result["bmm_result"]

    if np.allclose(torch_result_np, tf_result_np, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()