import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    eps = input_dict.get("eps", 1e-8)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.logit_(input_tensor, eps=eps)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-8)

        clipped_tensor = tf.clip_by_value(input_tensor, clip_value_min=eps, clip_value_max=1 - eps)
        result = tf.math.log(clipped_tensor / (1 - clipped_tensor))
        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        "eps": 1e-6
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.000001, 0.999999, 0.5], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32) * 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")


if __name__ == "__main__":
    main()