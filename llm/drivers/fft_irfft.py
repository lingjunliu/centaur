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

    result = torch.fft.irfft(input_tensor, n=n, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        n = input_dict.get("n", None)
        dim = input_dict.get("dim", -1)
        norm = input_dict.get("norm", "backward")

        input_tensor_complex = tf.cast(input_tensor, tf.complex128)
        
        if n is None:
            n = 2 * (tf.shape(input_tensor)[dim] - 1)
            n = int(n)

        fft_length = tf.constant([n], dtype=tf.int32)

        if norm == "forward":
            norm_factor = 1.0
        elif norm == "backward":
            norm_factor = float(1.0 / n)
        elif norm == "ortho":
            norm_factor = float(1.0 / np.sqrt(n))
        else:
            raise ValueError("Invalid norm value")

        result_complex = tf.signal.irfft(input_tensor_complex, fft_length=fft_length)[..., 0]
        result = result_complex * norm_factor
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1+0j, -0.4+0.3j, -0.3+0.0j, -0.4-0.3j], dtype=np.complex64),
        "n": 4,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()