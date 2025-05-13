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
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    bn_relu = nn.BatchNorm2d(num_features, eps, momentum, affine).float()
    bn_relu.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]).float(), requires_grad=False)
    bn_relu.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]).float(), requires_grad=False)
    bn_relu.running_mean = torch.tensor(input_dict["running_mean"]).float()
    bn_relu.running_var = torch.tensor(input_dict["running_var"]).float()
    bn_relu.eval()

    relu = nn.ReLU()
    
    result = relu(bn_relu(input_tensor.float()))

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)

        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)

        gamma = tf.reshape(weight, [1, 1, 1, num_features])
        beta = tf.reshape(bias, [1, 1, 1, num_features])
        mean = tf.reshape(running_mean, [1, 1, 1, num_features])
        variance = tf.reshape(running_var, [1, 1, 1, num_features])

        x = input_tensor
        x_normed = (x - mean) / tf.sqrt(variance + eps)
        x_scaled = gamma * x_normed + beta
        relu = tf.nn.relu(x_scaled)

        result = relu.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 1, 4, 4).astype(np.float32),
        "num_features": 1,
        "weight": np.random.rand(1).astype(np.float32),
        "bias": np.random.rand(1).astype(np.float32),
        "running_mean": np.random.rand(1).astype(np.float32),
        "running_var": np.random.rand(1).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()