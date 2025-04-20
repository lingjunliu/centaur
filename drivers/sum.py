import numpy as np

def torch_sum(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    keepdim = input.get("keepdim", False)
    dtype = torch.float32 if input.get("dtype", None) is None else torch.float64

    if dim is not None:
        sum_result = torch.sum(input_tensor, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        sum_result = torch.sum(input_tensor, dtype=dtype)

    if not cpu:
        sum_result = sum_result.cpu()

    return {"sum_result": sum_result.numpy()}

def tensorflow_sum(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        axis = input.get("dim", None)
        keepdims = input.get("keepdim", False)
        dtype = tf.float32 if input.get("dtype", None) is None else tf.float64

        if axis is not None:
            sum_result = tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
        else:
            sum_result = tf.reduce_sum(input_tensor)

        sum_result = tf.cast(sum_result, dtype)

        return {"sum_result": sum_result.numpy()}

def main():
    # Example 1: Sum all elements without dimension
    input_data_1 = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": None,
        "keepdim": False,
        "dtype": None,
    }

    # Example 2: Sum with dimension
    input_data_2 = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "keepdim": False,
        "dtype": None,
    }

    # PyTorch and TensorFlow examples
    for input_data in [input_data_1, input_data_2]:
        torch_result = torch_sum(input_data)
        print("Torch result:", torch_result)

        tf_result = tensorflow_sum(input_data)
        print("TensorFlow result:", tf_result)

        # Assert to check if the results are equal
        np.testing.assert_allclose(torch_result["sum_result"], tf_result["sum_result"], rtol=1e-5, atol=1e-8)
        if np.allclose(torch_result["sum_result"], tf_result["sum_result"], rtol=1e-5, atol=1e-8):
            print("equal")
        else:
            print("not equal")

if __name__ == "__main__":
    main()