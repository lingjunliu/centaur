import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.quantized as nniq

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    device = torch.device("cuda" if not cpu else "cpu")

    if not cpu:
        input_tensor = input_tensor.to(device)

    factory_kwargs = {'device': device} if not cpu else {}
    bn_relu = nniq.BNReLU3d(num_features, eps, momentum, affine, **factory_kwargs)

    if affine:
        bn_relu.weight = nn.Parameter(torch.tensor(input_dict["weight"], device=device))
        bn_relu.bias = nn.Parameter(torch.tensor(input_dict["bias"], device=device))
    
    bn_relu.running_mean = torch.tensor(input_dict["running_mean"], device=device)
    bn_relu.running_var = torch.tensor(input_dict["running_var"], device=device)

    result = bn_relu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)

        weight = tf.constant(input_dict["weight"], dtype=tf.float32) if affine else None
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if affine else None
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)

        mean = running_mean
        variance = running_var

        gamma = weight if affine else tf.ones([num_features], dtype=tf.float32)
        beta = bias if affine else tf.zeros([num_features], dtype=tf.float32)

        x = input_tensor
        
        def batch_norm_relu(x, mean, variance, beta, gamma, epsilon):
            normed = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)
            return tf.nn.relu(normed)

        result = batch_norm_relu(x, mean, variance, beta, gamma, eps)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "num_features": 3,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()