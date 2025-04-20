import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_matrix = torch.tensor(input["input"])
    vec = torch.tensor(input["vec"])

    # Perform matrix-vector multiplication
    result = torch.mv(input_matrix, vec)

    if not cpu:
        result = result.cpu()

    return {"mv_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_matrix = tf.constant(input["input"])
        vec = tf.constant(input["vec"])

        # Perform matrix-vector multiplication
        result = tf.linalg.matvec(input_matrix, vec)

        return {"mv_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "vec": np.array([7.0, 8.0, 9.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_mv_result = np.array(torch_result["mv_result"])
    tf_mv_result = np.array(tf_result["mv_result"])

    if np.allclose(torch_mv_result, tf_mv_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()