import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    window_length = int(input_dict["window_length"])
    periodic = input_dict.get("periodic", True)

    result = torch.blackman_window(window_length, periodic=periodic)

    if not cpu:
        result = result.cpu()

    return {"result": result.astype(np.float64)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    window_length = int(input_dict["window_length"])
    periodic = input_dict.get("periodic", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if window_length == 1:
            result = tf.ones((1,), dtype=tf.float64)
        else:
            M = window_length
            if not periodic:
                M = M + 1

            n = tf.cast(tf.range(M), dtype=tf.float64)
            result = 0.42 - 0.5 * tf.cos(2 * np.pi * n / (M - 1)) + 0.08 * tf.cos(4 * np.pi * n / (M - 1))

            if not periodic:
                result = result[:-1]

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "window_length": 10,
        "periodic": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    input_data = {
        "window_length": 10,
        "periodic": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()