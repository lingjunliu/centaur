import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.irfftn(input_tensor, s=s, dim=dim, norm=norm)

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
        input_tensor = tf.cast(tf.constant(input_dict["input"]), tf.complex64)
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", None)
        norm = input_dict.get("norm", "backward")

        if dim is None:
            dim = list(range(len(input_tensor.shape)))

        result = input_tensor
        norm_factor = 1.0

        for i, axis in enumerate(dim):
            fft_length = s[i]
            if isinstance(fft_length, tf.Tensor) or isinstance(fft_length, np.ndarray):
                fft_length = int(fft_length)

            result = tf.signal.irfft(result, fft_length=fft_length)
            result = tf.transpose(result, perm=get_transpose_permutation(axis, len(result.shape)))

        if norm == "forward":
            pass
        elif norm == "backward":
            n = np.prod(s)
            norm_factor = 1.0 / n
        elif norm == "ortho":
            n = np.prod(s)
            norm_factor = 1.0 / np.sqrt(n)

        result = result * norm_factor
        result = result.numpy()

    return {"result": result}

def get_transpose_permutation(axis, ndims):
    p = list(range(ndims))
    p.pop(axis)
    p = [axis] + p
    return p

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, -1, 1],
                           [-1, 1, -1],
                           [1, -1, 1]], dtype=np.float32),
        "s": (3, 4),
        "dim": (0, 1),
        "norm": "backward"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, -1, 1],
                           [-1, 1, -1],
                           [1, -1, 1]], dtype=np.float32),
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, -1, 1],
                           [-1, 1, -1],
                           [1, -1, 1]], dtype=np.float32),
        "s": (3, 4)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, -1, 1],
                           [-1, 1, -1],
                           [1, -1, 1]], dtype=np.float32),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()