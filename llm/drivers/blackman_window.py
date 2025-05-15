import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)

    result = torch.blackman_window(window_length, periodic=periodic)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if window_length == 1:
            result = tf.constant([1.0], dtype=tf.float32)
        else:
            if periodic:
                M = window_length
            else:
                M = window_length - 1

            n = tf.cast(tf.range(window_length, dtype=tf.float32), dtype=tf.float32)
            result = 0.5 - 0.5 * tf.cos(2 * np.pi * n / tf.cast(M - 1, dtype=tf.float32))

            if not periodic and window_length > 1:
                result = result[:-1]

            result = tf.cast(result, dtype=tf.float32)
            if window_length == 2 and not periodic:
                result = tf.constant([0.0, 0.0], dtype=tf.float32)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "window_length": 5,
        "periodic": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "window_length": 6,
        "periodic": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "window_length": 1,
        "periodic": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "window_length": 1,
        "periodic": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "window_length": 2,
        "periodic": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()