import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.ifft(input_tensor, n=n, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")
    
    input_tensor_rank = len(input_tensor.shape)
    dim_tf = dim
    if dim < 0:
        dim_tf = input_tensor_rank + dim

    if n is not None:
        input_shape = input_tensor.shape
        pad_shape = list(input_shape)
        pad_shape[dim_tf] = max(0, n - input_shape[dim_tf])
        if pad_shape[dim_tf] > 0:
            padding = [[0, 0] for _ in range(input_tensor_rank)]
            padding[dim_tf] = [0, pad_shape[dim_tf]]
            input_tensor = tf.pad(input_tensor, padding)
        elif n < input_shape[dim_tf]:
            slices = [slice(None) for _ in range(input_tensor_rank)]
            slices[dim_tf] = slice(0, n)
            input_tensor = input_tensor[tuple(slices)]
    
    input_tensor = tf.cast(input_tensor, tf.complex128)
    result = tf.signal.ifft(input_tensor)
    
    if norm == "forward":
        pass
    elif norm == "backward":
        if n is None:
            n = tf.cast(tf.shape(input_tensor)[dim_tf], tf.float64)
        else:
            n = tf.cast(n, tf.float64)
        result = result / tf.cast(n, tf.complex128)
    elif norm == "ortho":
        if n is None:
            n = tf.cast(tf.shape(input_tensor)[dim_tf], tf.float64)
        else:
            n = tf.cast(n, tf.float64)
        result = result / tf.cast(tf.sqrt(n), tf.complex128)

    result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j], dtype=np.complex64),
        "n": 16,
        "norm": 'ortho'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()