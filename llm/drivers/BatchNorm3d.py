import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch import nn

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if affine:
        weight = torch.tensor(input_dict["weight"])
        bias = torch.tensor(input_dict["bias"])
    if track_running_stats:
        running_mean = torch.tensor(input_dict["running_mean"])
        running_var = torch.tensor(input_dict["running_var"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if affine:
            weight = weight.cuda()
            bias = bias.cuda()
        if track_running_stats:
            running_mean = running_mean.cuda()
            running_var = running_var.cuda()

    batchnorm3d = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
    if affine:
        batchnorm3d.weight.data = weight
        batchnorm3d.bias.data = bias
    if track_running_stats:
        batchnorm3d.running_mean.data = running_mean
        batchnorm3d.running_var.data = running_var
    batchnorm3d.eval()

    if not cpu:
        batchnorm3d = batchnorm3d.cuda()
        
    with torch.no_grad():
        result = batchnorm3d(input_tensor)

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
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        if affine:
            weight = tf.constant(input_dict["weight"])
            bias = tf.constant(input_dict["bias"])
        if track_running_stats:
            running_mean = tf.constant(input_dict["running_mean"])
            running_var = tf.constant(input_dict["running_var"])

        if track_running_stats:
          mean = tf.reshape(running_mean, [1, num_features, 1, 1, 1])
          variance = tf.reshape(running_var, [1, num_features, 1, 1, 1])
        else:
          mean, variance = tf.nn.moments(input_tensor, axes=[0, 2, 3, 4], keepdims=True)

        inv = tf.math.rsqrt(variance + eps)
        if affine:
            inv *= tf.reshape(weight, [1, num_features, 1, 1, 1])
        output = input_tensor - mean
        output *= inv
        if affine:
            output += tf.reshape(bias, [1, num_features, 1, 1, 1])
        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "num_features": 3,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()