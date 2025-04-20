import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    tol = input.get("tol", None)
    hermitian = input.get("symmetric", False)

    # Apply to torch.linalg.matrix_rank
    if tol is not None:
        tol_tensor = torch.tensor(tol)
        rank = torch.linalg.matrix_rank(input_tensor, tol=tol_tensor, hermitian=hermitian)
    else:
        rank = torch.linalg.matrix_rank(input_tensor, tol=tol, hermitian=hermitian)

    if not cpu:
        rank = rank.cpu()

    return {"matrix_rank": int(rank.item())}

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
        tol = input.get("tol", None)

        # Apply to TensorFlow equivalent
        if tol is not None:
            tol_tensor = tf.constant(tol)
            s = tf.linalg.svd(input_tensor, compute_uv=False)
            rank = tf.reduce_sum(tf.cast(s > tol_tensor, tf.int32))
        else:
            s = tf.linalg.svd(input_tensor, compute_uv=False)
            rank = tf.reduce_sum(tf.cast(s > 0, tf.int32))

        return {"matrix_rank": int(rank.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # Example matrix
        "tol": 1e-5,
        "symmetric": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if torch_result["matrix_rank"] == tf_result["matrix_rank"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()