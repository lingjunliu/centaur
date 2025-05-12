import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["y"])
    x = input_dict.get("x", None)
    if x is not None:
        x = torch.tensor(x)
    dim = input_dict.get("dim", -1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if x is not None:
            x = x.cuda()

    result = torch.cumulative_trapezoid(y=input_tensor, x=x, dim=dim)

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
        y = tf.constant(input_dict["y"])
        x = input_dict.get("x", None)
        if x is not None:
            x = tf.constant(x)
        dim = input_dict.get("dim", -1)

        rank = len(y.shape)
        if dim < 0:
            dim = rank + dim + 1

        def trapezoid(y, x=None, dim=-1):
            if x is None:
                delta = tf.ones_like(y, dtype=y.dtype)
                if len(y.shape) > 1:
                    delta = delta[(slice(None), slice(1,None)) if dim==1 else (slice(1,None), slice(None))]
                else:
                    delta = delta[1:]
            else:
                delta = tf.experimental.numpy.diff(x, axis=dim)

            if len(y.shape) > 1:
                y1 = y[(slice(None), slice(None, -1)) if dim==1 else (slice(None, -1), slice(None))]
                y2 = y[(slice(None), slice(1, None)) if dim==1 else (slice(1, None), slice(None))]
            else:
                y1 = y[slice(None, -1)]
                y2 = y[slice(1, None)]

            integral = tf.math.cumsum(delta * (y1 + y2) / 2., axis=dim)

            padding_shape = [[0,0]] * rank
            padding_shape[dim] = [1, 0]

            zeros_shape = list(y.shape)
            zeros_shape[dim] = 1
            zeros = tf.zeros(zeros_shape, dtype=y.dtype)
            integral = tf.concat([zeros, integral], axis=dim)
            return integral

        result = trapezoid(y=y, x=x, dim=dim)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "y": np.array([1, 2, 3, 4], dtype=np.float32),
        "x": np.array([2, 4, 6, 8], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    min_len = min(len(torch_result["result"]), len(tf_result["result"]))
    assert np.allclose(torch_result["result"][:min_len], tf_result["result"][:min_len], atol=A_TOL), "Results do not match"

    input_data2 = {
        "y": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "dim": 1
    }

    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)
    assert np.allclose(torch_result2["result"], tf_result2["result"], atol=A_TOL), "Results do not match"

    input_data3 = {
        "y": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "x": np.array([1,2,3], dtype=np.float32),
        "dim": 1
    }

    torch_result3 = torch_version(input_data3)
    tf_result3 = tensorflow_version(input_data3)

    assert np.allclose(torch_result3["result"], tf_result3["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()