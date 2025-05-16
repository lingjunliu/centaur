import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    n = input_dict["n"]
    d = input_dict.get("d", 1.0)
    requires_grad = input_dict.get("requires_grad", False)
    dtype = input_dict.get("dtype", None)

    if not cpu:
        torch.set_default_device('cuda')

    if dtype is not None:
        result = torch.fft.fftfreq(n, d=d, dtype=getattr(torch, dtype), requires_grad=requires_grad)
    else:
        result = torch.fft.fftfreq(n, d=d, requires_grad=requires_grad)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    n = input_dict["n"]
    d = input_dict.get("d", 1.0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if n % 2 == 0:
            positive_freqs = tf.range(0, n // 2)
            negative_freqs = tf.range(-n // 2, 0)
            freqs = tf.concat([positive_freqs, negative_freqs], axis=0)
        else:
            positive_freqs = tf.range(0, (n - 1) // 2 + 1)
            negative_freqs = tf.range(-(n - 1) // 2, 0)
            freqs = tf.concat([positive_freqs, negative_freqs], axis=0)

        result = tf.cast(freqs, dtype=tf.float32) / (d * n)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "n": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": 4
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": 4,
        "d": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()