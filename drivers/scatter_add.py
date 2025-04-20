import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Convert inputs to torch tensors
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    index_tensor = torch.tensor(input["index"], dtype=torch.int64)
    src_tensor = torch.tensor(input["src"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        src_tensor = src_tensor.cuda()
        index_tensor = index_tensor.cuda()

    # Perform scatter_add operation
    result = torch.scatter_add(input_tensor, dim, index_tensor, src_tensor)

    if not cpu:
        result = result.cpu()

    return {"scatter_add_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert inputs to tensorflow tensors
        input_tensor = tf.constant(input["input"])
        index_tensor = tf.constant(input["index"], dtype=tf.int32)
        src_tensor = tf.constant(input["src"])
        dim = input["dim"]

        # Perform scatter_add operation
        if dim == 0:
            indices = tf.stack([index_tensor,
                                tf.tile(tf.range(input_tensor.shape[1], dtype=tf.int32)[tf.newaxis, :],
                                        [index_tensor.shape[0], 1])], axis=2)
        elif dim == 1:
            indices = tf.stack([tf.tile(tf.range(input_tensor.shape[0], dtype=tf.int32)[:, tf.newaxis],
                                        [1, index_tensor.shape[1]]),
                                index_tensor], axis=2)
        else:
            raise ValueError("Currently only supporting dim=0 and dim=1")

        # Reshape indices to [N, 2]
        indices_flat = tf.reshape(indices, [-1, 2])
        src_flat = tf.reshape(src_tensor, [-1])

        result = tf.tensor_scatter_nd_add(input_tensor, indices_flat, src_flat)

        return {"scatter_add_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "dim": 1,
        "index": np.array([[0, 1, 2], [0, 1, 1]], dtype=np.int64),
        "src": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result["scatter_add_result"], tf_result["scatter_add_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
