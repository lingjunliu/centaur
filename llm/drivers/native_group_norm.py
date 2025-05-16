import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    num_groups = input_dict["num_groups"]
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    eps = input_dict.get("eps", 1e-5)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.nn.functional.group_norm(input_tensor, num_groups, weight, bias, eps)

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
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-5)

        shape = tf.shape(input_tensor)
        N = shape[0]
        C = shape[1]
        H = shape[2]
        W = shape[3]

        x = tf.reshape(input_tensor, [N, num_groups, C // num_groups, H, W])

        mean, variance = tf.nn.moments(x, axes=[2, 3, 4], keepdims=True)

        x = (x - mean) / tf.sqrt(variance + eps)

        x = tf.reshape(x, [N, C, H, W])

        result = weight[None, :, None, None] * x + bias[None, :, None, None]
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(2, 6, 4, 4).astype(np.float32),
        "num_groups": 3,
        "weight": np.random.randn(6).astype(np.float32),
        "bias": np.random.randn(6).astype(np.float32),
        "eps": 1e-5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()