import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    module = nn.BatchNorm2d(num_features)

    module.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
    module.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))
    module.running_mean = torch.tensor(input_dict["running_mean"])
    module.running_var = torch.tensor(input_dict["running_var"])
    module.eps = input_dict.get("eps", 1e-05)
    module.momentum = input_dict.get("momentum", 0.1)

    if not cpu:
        module.cuda()
        module.weight = torch.nn.Parameter(module.weight.cuda())
        module.bias = torch.nn.Parameter(module.bias.cuda())
        module.running_mean = module.running_mean.cuda()
        module.running_var = module.running_var.cuda()

    result = module(input_tensor)
    result = torch.relu(result)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        num_features = input_dict["num_features"]
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)

        weight = tf.reshape(weight, [1, num_features, 1, 1])
        bias = tf.reshape(bias, [1, num_features, 1, 1])
        running_mean = tf.reshape(running_mean, [1, num_features, 1, 1])
        running_var = tf.reshape(running_var, [1, num_features, 1, 1])

        result = tf.nn.batch_normalization(
            input_tensor,
            mean=running_mean,
            variance=running_var,
            offset=bias,
            scale=weight,
            variance_epsilon=eps
        )
        result = tf.nn.relu(result)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "num_features": 3,
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()