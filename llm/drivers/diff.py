import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", 1)
    dim = input_dict.get("dim", -1)
    prepend = input_dict.get("prepend", None)
    append = input_dict.get("append", None)

    if prepend is not None:
        prepend = torch.tensor(prepend)
    if append is not None:
        append = torch.tensor(append)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if prepend is not None:
            prepend = prepend.cuda()
        if append is not None:
            append = append.cuda()

    result = torch.diff(input_tensor, n=n, dim=dim, prepend=prepend, append=append)

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
        n = input_dict.get("n", 1)
        dim = input_dict.get("dim", -1)
        prepend = input_dict.get("prepend", None)
        append = input_dict.get("append", None)

        if prepend is not None:
            prepend = tf.constant(prepend)
        if append is not None:
            append = tf.constant(append)

        def tf_diff(x, n, dim, prepend, append):
            if prepend is not None:
                x = tf.concat([prepend, x], axis=dim)
            if append is not None:
                x = tf.concat([x, append], axis=dim)

            if n == 0:
                return x
            else:
                slices = [slice(None)] * len(x.shape)
                slices[dim] = slice(1, None)
                slices_minus_one = [slice(None)] * len(x.shape)
                slices_minus_one[dim] = slice(None, -1)
                diff = tf.gather(x, list(range(1, x.shape[dim])), axis=dim) - tf.gather(x, list(range(x.shape[dim]-1)), axis=dim)

                return tf_diff(diff, n-1, dim, None, None)

        result = tf_diff(input_tensor, n, dim, prepend, append)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 3, 2], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 3, 2], dtype=np.float32),
        "append": np.array([4, 5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "n": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "prepend": np.array([0], dtype=np.float32),
        "append": np.array([6], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()