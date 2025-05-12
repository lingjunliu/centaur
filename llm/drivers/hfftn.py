import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.hfftn(input_tensor, s=s, dim=dim, norm=norm)

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
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", None)
        norm = input_dict.get("norm", "backward")

        if input_tensor.dtype != tf.complex64 and input_tensor.dtype != tf.complex128:
             input_tensor_complex = tf.complex(tf.cast(tf.math.real(input_tensor), tf.float32), tf.zeros_like(tf.math.real(input_tensor), dtype=tf.float32))
        else:
            input_tensor_complex = input_tensor

        rank = len(input_tensor.shape)
        if dim is None:
            dim = tuple(range(rank))

        fft_length = None
        if s is not None:
            fft_length = s

        result = tf.signal.fft(tf.cast(input_tensor_complex, tf.complex64))

        if norm == "forward":
            if fft_length is not None:
              n = tf.cast(tf.reduce_prod([fft_length[i] for i in range(len(fft_length))]), tf.float32)
            else:
              n = tf.cast(tf.reduce_prod([tf.shape(input_tensor)[i] for i in dim]), tf.float32)
            result = result / n
        elif norm == "ortho":
            if fft_length is not None:
              n = tf.cast(tf.reduce_prod([fft_length[i] for i in range(len(fft_length))]), tf.float32)
            else:
              n = tf.cast(tf.reduce_prod([tf.shape(input_tensor)[i] for i in dim]), tf.float32)
            result = result / tf.sqrt(n)

        result = tf.math.real(result).numpy()

        if s is not None:
            s = tuple(s)
        else:
            s = tuple(input_tensor.shape.as_list())

        min_rank = min(len(s), len(result.shape))
        slices = tuple(slice(0, min(s[i], result.shape[i])) for i in range(min_rank))

        while len(slices) < len(result.shape):
            slices = slices + (slice(0, result.shape[len(slices)]),)
            
        result = result[slices]
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.+0.j, 2.+0.j, 3.+0.j], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1.+0.j, 2.+0.j, 3.+0.j], [4.+0.j, 5.+0.j, 6.+0.j]], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1.+0.j, 2.+0.j, 3.+0.j], [4.+0.j, 5.+0.j, 6.+0.j]], dtype=np.complex64),
        "s": (2, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j], dtype=np.complex64),
        "s": (3,)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()