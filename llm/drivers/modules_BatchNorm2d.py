import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])

    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    bn = torch.nn.BatchNorm2d(input_tensor.shape[1])
    bn.running_mean = running_mean
    bn.running_var = running_var
    bn.weight = torch.nn.Parameter(weight)
    bn.bias = torch.nn.Parameter(bias)
    bn.momentum = momentum
    bn.eps = eps
    bn.train(training)

    result = bn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    running_mean = tf.constant(input_dict["running_mean"])
    running_var = tf.constant(input_dict["running_var"])
    weight = tf.constant(input_dict["weight"])
    bias = tf.constant(input_dict["bias"])

    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    training = input_dict.get("training", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        num_channels = input_tensor.shape[1]
        
        weight = tf.reshape(weight, [1, num_channels, 1, 1])
        bias = tf.reshape(bias, [1, num_channels, 1, 1])
        running_mean = tf.reshape(running_mean, [1, num_channels, 1, 1])
        running_var = tf.reshape(running_var, [1, num_channels, 1, 1])

        if training:
            axes = [0, 2, 3]
            mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

            running_mean_new = running_mean * (1 - momentum) + mean * momentum
            running_var_new = running_var * (1 - momentum) + variance * momentum

            def training_fn():
              return tf.nn.batch_normalization(input_tensor, mean, variance, offset=bias, scale=weight, variance_epsilon=eps)

            def non_training_fn():
              return tf.nn.batch_normalization(input_tensor, running_mean, running_var, offset=bias, scale=weight, variance_epsilon=eps)

            result = training_fn()

        else:
            result = tf.nn.batch_normalization(input_tensor, running_mean, running_var, offset=bias, scale=weight, variance_epsilon=eps)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(2, 3, 4, 4).astype(np.float32),
        "running_mean": np.random.randn(3).astype(np.float32),
        "running_var": np.abs(np.random.randn(3)).astype(np.float32),
        "weight": np.random.randn(3).astype(np.float32),
        "bias": np.random.randn(3).astype(np.float32),
        "momentum": 0.1,
        "eps": 1e-05,
        "training": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()