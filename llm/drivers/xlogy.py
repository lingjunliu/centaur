import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    x = torch.tensor(input_dict["x"])
    y = torch.tensor(input_dict["y"])

    if not cpu:
        x = x.cuda()
        y = y.cuda()

    result = torch.special.xlogy(x, y)

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
        x = tf.constant(input_dict["x"], dtype=tf.float32)
        y = tf.constant(input_dict["y"], dtype=tf.float32)

        result = tf.math.xlogy(x, y)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "y": np.array([1.0, 0.0, -1.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()