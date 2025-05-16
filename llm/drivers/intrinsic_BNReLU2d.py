import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic as nni

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict.get("num_features")
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    bn_relu = nni.BNReLU2d(num_features)
    bn_relu.bn.eps = eps
    bn_relu.bn.momentum = momentum
    bn_relu.bn.affine = affine
    bn_relu.bn.track_running_stats = track_running_stats
    bn_relu.bn.weight = nn.Parameter(torch.tensor(input_dict["weight"]))
    bn_relu.bn.bias = nn.Parameter(torch.tensor(input_dict["bias"]))
    bn_relu.bn.running_mean = torch.tensor(input_dict["running_mean"])
    bn_relu.bn.running_var = torch.tensor(input_dict["running_var"])
    bn_relu.bn.num_batches_tracked = torch.tensor(input_dict["num_batches_tracked"])
    
    if not cpu:
        bn_relu = bn_relu.cuda()
        bn_relu.bn.weight = nn.Parameter(bn_relu.bn.weight.cuda())
        bn_relu.bn.bias = nn.Parameter(bn_relu.bn.bias.cuda())
        bn_relu.bn.running_mean = bn_relu.bn.running_mean.cuda()
        bn_relu.bn.running_var = bn_relu.bn.running_var.cuda()
        bn_relu.bn.num_batches_tracked = bn_relu.bn.num_batches_tracked.cuda()

    result = bn_relu(input_tensor)

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
        num_features = input_dict.get("num_features")
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])

        mean, variance = tf.nn.moments(input_tensor, axes=[0, 2, 3], keepdims=True)

        scale = weight
        offset = bias

        inv = tf.math.rsqrt(running_var + eps)
        normalized = (input_tensor - running_mean) * inv
        if affine:
            output = scale * normalized + offset
        else:
            output = normalized
        output = tf.nn.relu(output)

        result = output.numpy()

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
        "num_batches_tracked": np.array(0).astype(np.int64),
        "affine": True,
        "track_running_stats": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()