import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    num_groups = input_dict["num_groups"]
    num_channels = input_dict["num_channels"]
    eps = input_dict.get("eps", 1e-05)
    affine = input_dict.get("affine", True)

    if affine:
        weight = torch.tensor(input_dict["weight"])
        bias = torch.tensor(input_dict["bias"])
    else:
        weight = None
        bias = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        if affine:
            weight = weight.cuda()
            bias = bias.cuda()

    norm = torch.nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
    if affine:
        norm.weight.data = weight
        norm.bias.data = bias

    result = norm(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        num_groups = input_dict["num_groups"]
        num_channels = input_dict["num_channels"]
        eps = input_dict.get("eps", 1e-05)
        affine = input_dict.get("affine", True)

        if affine:
            weight = tf.constant(input_dict["weight"])
            bias = tf.constant(input_dict["bias"])
        else:
            weight = None
            bias = None

        shape = tf.shape(input_tensor)
        N = shape[0]
        H = shape[2]
        W = shape[3]

        x = tf.reshape(input_tensor, [N, num_groups, num_channels // num_groups, H, W])

        mean, variance = tf.nn.moments(x, axes=[2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + eps)

        x = tf.reshape(x, [N, num_channels, H, W])

        if affine:
            x = x * tf.reshape(weight, [1, num_channels, 1, 1]) + tf.reshape(bias, [1, num_channels, 1, 1])

        result = x.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 6, 4, 4).astype(np.float32),
        "num_groups": 3,
        "num_channels": 6,
        "weight": np.random.rand(6).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()