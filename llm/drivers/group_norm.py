import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    num_groups = input_dict["num_groups"]
    num_channels = input_dict["num_channels"]
    eps = input_dict.get("eps", 1e-5)
    affine = input_dict.get("affine", True)
    if affine:
        weight = torch.tensor(input_dict["weight"])
        bias = torch.tensor(input_dict["bias"])
    else:
        weight = None
        bias = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        if affine and weight is not None and bias is not None:
            weight = weight.cuda()
            bias = bias.cuda()

    if affine and weight is not None and bias is not None:
        result = torch.group_norm(input_tensor, num_groups, num_channels, weight, bias, eps)
    else:
        result = torch.group_norm(input_tensor, num_groups, num_channels, None, None, eps)

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
        input_tensor = tf.constant(input_dict["input"])
        num_groups = input_dict["num_groups"]
        num_channels = input_dict["num_channels"]
        eps = input_dict.get("eps", 1e-5)
        affine = input_dict.get("affine", True)

        shape = tf.shape(input_tensor)
        N = shape[0]
        C = num_channels
        H = shape[2]
        W = shape[3]

        x = tf.reshape(input_tensor, [N, num_channels, H, W])
        group_shape = tf.shape(x)
        x = tf.reshape(x, [N, num_groups, C // num_groups, H, W])

        mean, variance = tf.nn.moments(x, axes=[2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(tf.maximum(variance, 0.0) + eps)
        x = tf.reshape(x, [N, C, H, W])

        if affine:
            weight = tf.constant(input_dict["weight"], dtype=tf.float32)
            bias = tf.constant(input_dict["bias"], dtype=tf.float32)
            result = x * tf.reshape(weight, [1, C, 1, 1]) + tf.reshape(bias, [1, C, 1, 1])
        else:
            result = x

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 6, 4, 4).astype(np.float32),
        "num_groups": 3,
        "num_channels": 6,
        "eps": 1e-5,
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