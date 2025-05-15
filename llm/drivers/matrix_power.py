import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict["n"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.linalg.matrix_power(input_tensor, n)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        n = input_dict["n"]

        input_tensor_shape = input_tensor.shape
        if len(input_tensor_shape) != 2 or input_tensor_shape[0] != input_tensor_shape[1]:
            raise ValueError("Input must be a square matrix.")

        if n == 0:
            result = tf.eye(input_tensor_shape[0], dtype=input_tensor.dtype)
        elif n > 0:
            result = input_tensor
            for _ in range(n - 1):
                result = tf.matmul(result, input_tensor)
        else:
            input_tensor_inv = tf.linalg.inv(input_tensor)
            result = input_tensor_inv
            for _ in range(-n - 1):
                result = tf.matmul(result, input_tensor_inv)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "n": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "n": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "n": -1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()