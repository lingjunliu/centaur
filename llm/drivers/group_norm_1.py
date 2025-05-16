import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    num_groups = input_dict["num_groups"]
    num_channels = input_dict["num_channels"]
    eps = input_dict.get("eps", 1e-05)
    affine = input_dict["affine"]
    weight = input_dict.get("weight", None)
    bias = input_dict.get("bias", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if weight is not None:
            weight = torch.tensor(weight).cuda()
        if bias is not None:
            bias = torch.tensor(bias).cuda()

    if weight is not None:
        weight = torch.tensor(weight)
        if not cpu:
             weight = weight.cuda()
    if bias is not None:
        bias = torch.tensor(bias)
        if not cpu:
             bias = bias.cuda()

    result = torch.nn.functional.group_norm(input_tensor, num_groups, num_channels, weight=weight, bias=bias, eps=eps)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        num_groups = input_dict["num_groups"]
        num_channels = input_dict["num_channels"]
        eps = input_dict.get("eps", 1e-05)
        affine = input_dict["affine"]
        weight = input_dict.get("weight", None)
        bias = input_dict.get("bias", None)

        input_shape = input_tensor.shape
        N = input_shape[0] if len(input_shape) > 1 else 1
        C = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        H = input_shape[2] if len(input_shape) > 2 else 1
        W = input_shape[3] if len(input_shape) > 3 else 1

        x = tf.reshape(input_tensor, [N, num_groups, num_channels // num_groups, H, W])
        mean, variance = tf.nn.moments(x, axes=[2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + eps)
        x = tf.reshape(x, [N, C, H, W])

        if affine:
            if weight is not None:
                gamma = tf.constant(weight, dtype=tf.float32)
            else:
                gamma = tf.ones(num_channels, dtype=tf.float32)

            if bias is not None:
                beta = tf.constant(bias, dtype=tf.float32)
            else:
                beta = tf.zeros(num_channels, dtype=tf.float32)

            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])
            x = gamma * x + beta

        result = x.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 6, 4, 4).astype(np.float32),
        "num_groups": 3,
        "num_channels": 6,
        "eps": 1e-05,
        "affine": True,
        "weight": np.random.rand(6).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()