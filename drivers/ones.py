import numpy as np


def torch_ones_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    size = tuple(input["size"])
    dtype = torch.float32 if input.get("dtype", None) is None else input["dtype"]
    device = torch.device("cpu") if cpu else torch.device("cuda")

    # Create tensor filled with ones using torch.ones
    tensor = torch.ones(size, dtype=dtype, device=device)

    if not cpu:
        tensor = tensor.cpu()

    return {"ones_tensor": tensor.numpy()}


def tensorflow_ones_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    size = input["size"]
    dtype = tf.float32 if input.get("dtype", None) is None else input["dtype"]
    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Create TensorFlow variable filled with ones using tf.ones
        tensor = tf.ones(shape=size, dtype=dtype)

    return {"ones_tensor": tensor.numpy()}


def main():
    # Example input
    input_data = {
        "size": [2, 3],
        "dtype": None
    }

    # Torch example
    torch_result = torch_ones_version(input_data)
    print("Torch result:\n", torch_result["ones_tensor"])

    # TensorFlow example
    tf_result = tensorflow_ones_version(input_data)
    print("TensorFlow result:\n", tf_result["ones_tensor"])

    assert np.array_equal(torch_result["ones_tensor"], tf_result["ones_tensor"]), "The outputs are not equal"
    print("equal")


if __name__ == "__main__":
    main()