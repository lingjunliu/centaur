import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.ZeroPad1d(padding)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = input_tensor.shape
        input_rank = len(input_shape)
        if input_rank == 1:
            padded_tensor = tf.pad(input_tensor, [[padding[0], padding[1]]], "CONSTANT")
        elif input_rank == 2:
            padded_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[1]]], "CONSTANT")
        elif input_rank == 3:
            padded_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[1]], [0, 0]], "CONSTANT")
        else:
            raise ValueError("Input tensor must have rank 1, 2, or 3")

        result = padded_tensor.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "padding": (2, 3),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0],[5,6,7,8]], dtype=np.float32),
        "padding": (2, 3),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()