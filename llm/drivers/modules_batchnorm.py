import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.nn.functional.batch_norm(
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
    )

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
        input_tensor = tf.constant(input_dict["input"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        if training:
            mean, variance = tf.nn.moments(input_tensor, axes=[0])
            result = tf.nn.batch_normalization(
                input_tensor, mean, variance, bias, weight, eps
            )

            new_running_mean = running_mean * momentum + mean * (1 - momentum)
            new_running_var = running_var * momentum + variance * (1 - momentum)

        else:
            result = tf.nn.batch_normalization(
                input_tensor, running_mean, running_var, bias, weight, eps
            )

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array(
            [[0.0202, 1.0985], [1.3506, -0.6056], [0.0202, 1.0985], [1.3506, -0.6056]],
            dtype=np.float32,
        ),
        "running_mean": np.array([0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0], dtype=np.float32),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-05,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(
        torch_result["result"], tf_result["result"], atol=A_TOL
    ), "Results do not match"

    input_data["training"] = True
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print("Success")

if __name__ == "__main__":
    main()