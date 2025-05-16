import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.complex64)
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.irfft(input_tensor, n=n, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.complex64)
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        
        rank = len(input_tensor.shape)
        dim = dim % rank
        
        if n is None:
            n = 2 * (tf.shape(input_tensor)[dim] - 1)
        
        fft_length = tf.cast(n, tf.int32)

        result = tf.signal.irfft(input_tensor, fft_length=[fft_length])

        if norm == "forward":
            norm_factor = 1.0
        elif norm == "backward":
            norm_factor = 1.0 / tf.cast(n, tf.float32)
        elif norm == "ortho":
            norm_factor = 1.0 / tf.sqrt(tf.cast(n, tf.float32))
        else:
            raise ValueError("Invalid norm value")
        
        result = result * norm_factor
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([2.5 + 0j, -0.625 + 0.8602j, -0.625 + 0.2031j], dtype=np.complex64),
        "n": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()