import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    window_length = torch.tensor(input_dict["window_length"])
    periodic = input_dict.get("periodic", True)
    beta = input_dict.get("beta", 5.0)

    if not cpu:
        window_length = window_length.cuda()

    result = torch.kaiser_window(window_length=int(window_length.item()), periodic=periodic, beta=beta)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        window_length = tf.constant(input_dict["window_length"], dtype=tf.int32)
        periodic = input_dict.get("periodic", True)
        beta = input_dict.get("beta", 5.0)
        beta = tf.cast(beta, dtype=tf.float32)

        result = tf.signal.kaiser_window(window_length, beta=beta)

        if not periodic:
            result = result[:-1]
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "window_length": 10,
        "periodic": True,
        "beta": 10.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()