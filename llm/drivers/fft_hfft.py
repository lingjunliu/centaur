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
    
    result = torch.fft.hfft(input_tensor, n=n, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm_str = input_dict.get("norm", "backward")

    if dim != -1:
        input_tensor = tf.transpose(input_tensor, perm=[dim] + [i for i in range(len(input_tensor.shape)) if i != dim])

    if input_tensor.dtype not in [tf.complex64, tf.complex128]:
        input_tensor = tf.complex(tf.cast(tf.math.real(input_tensor), tf.float32), tf.cast(tf.math.imag(input_tensor), tf.float32))
    
    input_size = input_tensor.shape[0]
    if n is None:
        n = 2 * (input_size - 1)

    N = tf.cast(n, dtype=tf.float32)

    hermitian_part = input_tensor

    full_fft_input = tf.concat([hermitian_part, tf.reverse(tf.math.conj(hermitian_part[1:-1]), axis=[0])], axis=0)
    
    if full_fft_input.shape[0] > n:
        fft_input = full_fft_input[:n]
    elif full_fft_input.shape[0] < n:
        fft_input = tf.pad(full_fft_input, [[0, n - full_fft_input.shape[0]]])
    else:
        fft_input = full_fft_input

    fft_result = tf.signal.fft(tf.cast(fft_input, tf.complex128))
    
    if norm_str == "forward":
        fft_result = fft_result / N
    elif norm_str == "ortho":
        fft_result = fft_result / tf.sqrt(N)
    
    if dim != -1:
        fft_result = tf.transpose(fft_result, perm=[i for i in range(1, len(fft_result.shape))] + [0])
    
    result = tf.cast(tf.math.real(fft_result), tf.float32).numpy()
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.5 - 0j, 0.125 - 0.17204989j, -0.125 - 0.04061497j], dtype=np.complex64),
        "n": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5 - 0j, 0.125 - 0.17204989j, -0.125 - 0.04061497j], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5 - 0j, 0.125 - 0.17204989j, -0.125 - 0.04061497j], dtype=np.complex64),
        "norm": "forward",
        "n": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.5 - 0j, 0.125 - 0.17204989j, -0.125 - 0.04061497j],
                          [0.5 - 0j, 0.125 - 0.17204989j, -0.125 - 0.04061497j]], dtype=np.complex64),
        "n": 5,
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()