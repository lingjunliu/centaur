import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict.get("num_features")
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    bn = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    if "running_mean" in input_dict:
        bn.running_mean = torch.tensor(input_dict["running_mean"])
    if "running_var" in input_dict:
        bn.running_var = torch.tensor(input_dict["running_var"])
    if "weight" in input_dict:
        bn.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
    if "bias" in input_dict:
        bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))

    if not cpu:
        bn = bn.cuda()

    result = bn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        num_features = input_dict.get("num_features")
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        gamma = None
        beta = None
        moving_mean = None
        moving_variance = None

        if "weight" in input_dict and affine:
            gamma = tf.constant(input_dict["weight"])
        if "bias" in input_dict and affine:
            beta = tf.constant(input_dict["bias"])
        if "running_mean" in input_dict and track_running_stats:
            moving_mean = tf.constant(input_dict["running_mean"])
        if "running_var" in input_dict and track_running_stats:
            moving_variance = tf.constant(input_dict["running_var"])

        input_shape = input_tensor.shape
        num_channels = num_features
        data_format = 'NHWC'
        axes = [0, 1, 2]

        if gamma is not None:
            # gamma = tf.reshape(gamma, (1, 1, 1, num_channels))
            pass
        if beta is not None:
            # beta = tf.reshape(beta, (1, 1, 1, num_channels))
            pass
        if moving_mean is not None:
            # moving_mean = tf.reshape(moving_mean, (1, 1, 1, num_channels))
            pass
        if moving_variance is not None:
            # moving_variance = tf.reshape(moving_variance, (1, 1, 1, num_channels))
            pass

        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)

        if not track_running_stats:
            result = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)
        else:
            if "running_mean" not in input_dict or "running_var" not in input_dict:

                result = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)

            else:
                result = tf.nn.batch_normalization(input_tensor, moving_mean, moving_variance, beta, gamma, eps)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 4, 4, 3).astype(np.float32),
        "num_features": 3,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
    }

    input_data["weight"] = np.random.rand(input_data["num_features"]).astype(np.float32)
    input_data["bias"] = np.random.rand(input_data["num_features"]).astype(np.float32)
    input_data["running_mean"] = np.random.rand(input_data["num_features"]).astype(np.float32)
    input_data["running_var"] = np.random.rand(input_data["num_features"]).astype(np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()