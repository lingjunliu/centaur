import numpy as np


def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["tensor"])

    # Move tensors to CPU or GPU
    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Apply to torch.dot
    result = torch.dot(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"dot_product": float(result.item())}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["tensor"])

        # Apply to TensorFlow equivalent
        result = tf.tensordot(input_tensor, other_tensor, axes=1)

        return {"dot_product": float(result.numpy())}


def main():
    # Example input
    input_data = {
        "input": np.array([2, 3], dtype=np.float32),
        "tensor": np.array([2, 1], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert results to a common format and assert equality
    if np.isclose(torch_result["dot_product"], tf_result["dot_product"]):
        print("equal")
    else:
        print("not equal")


if __name__ == "__main__":
    main()