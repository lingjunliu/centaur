import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)
    alpha = input_dict.get("alpha", 0.5)
    beta = input_dict.get("beta", 0.5)

    if not cpu:
        torch.set_default_device("cuda")

    result = torch.hamming_window(window_length, periodic=periodic, alpha=alpha, beta=beta)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)
    alpha = input_dict.get("alpha", 0.5)
    beta = input_dict.get("beta", 0.5)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        window_length_tf = tf.cast(window_length, tf.float32)

        if periodic:
            n = window_length_tf
        else:
            n = window_length_tf - 1

        window = tf.range(0, window_length_tf, dtype=tf.float32)
        result = alpha - beta * tf.cos(2 * np.pi * window / (n))

        if not periodic:
            result = result[:window_length]

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "window_length": 10,
        "periodic": True,
        "alpha": 0.5,
        "beta": 0.5,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()