import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.fft(input_tensor, n=n, dim=dim, norm=norm)

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
        n = input_dict.get("n", None)
        dim = input_dict.get("dim", -1)
        norm = input_dict.get("norm", None)

        rank = len(input_tensor.shape)
        dim = dim if dim >= 0 else rank + dim

        if n is None:
            n = tf.shape(input_tensor)[dim]
        
        input_tensor = tf.cast(input_tensor, tf.complex64)

        if n != tf.shape(input_tensor)[dim]:
            pad_size = n - tf.shape(input_tensor)[dim]
            if pad_size > 0:
                padding = [[0, 0]] * rank
                padding[dim] = [0, pad_size]
                input_tensor = tf.pad(input_tensor, padding)
            else:
                slice_indices = [slice(None)] * rank
                slice_indices[dim] = slice(0, n)
                input_tensor = input_tensor[tuple(slice_indices)]

        result = tf.signal.fft(input_tensor)

        if norm == "ortho":
            result = result / tf.cast(tf.sqrt(tf.cast(tf.cast(n, tf.float32),tf.complex64)), tf.complex64)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "n": 4,
        "dim": 0,
        "norm": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "n": 2,
        "dim": 1,
        "norm": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "n": 4,
        "dim": 1,
        "norm": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "n": 4,
        "dim": 0,
        "norm": 'ortho'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()