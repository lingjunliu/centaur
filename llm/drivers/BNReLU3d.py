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
    norm_weight = torch.tensor(input_dict["norm_weight"], dtype=torch.float32) if affine else None
    norm_bias = torch.tensor(input_dict["norm_bias"], dtype=torch.float32) if affine else None
    freeze_bn = input_dict.get("freeze_bn", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if affine:
            norm_weight = norm_weight.cuda()
            norm_bias = norm_bias.cuda()

    bn_relu = nniq.BNReLU3d(num_features, eps, momentum, affine)

    if affine:
        bn_relu.weight = nn.Parameter(norm_weight)
        bn_relu.bias = nn.Parameter(norm_bias)
    
    if freeze_bn:
        bn_relu.weight.requires_grad_(False)
        bn_relu.bias.requires_grad_(False)

    bn_relu.eval()
    with torch.no_grad():
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
        input_tensor = tf.constant(input_dict["input"])
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        norm_weight = tf.constant(input_dict["norm_weight"]) if affine else None
        norm_bias = tf.constant(input_dict["norm_bias"]) if affine else None
        freeze_bn = input_dict.get("freeze_bn", False)

        mean = tf.zeros([num_features])
        variance = tf.ones([num_features])

        input_shape = tf.shape(input_tensor)
        spatial_dims = len(input_shape) - 2
        reduction_axes = list(range(1, spatial_dims + 1))
        
        if affine:
            bn = tf.nn.batch_normalization(input_tensor, mean, variance, norm_bias, norm_weight, eps)
        else:
            bn = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, eps)
        
        relu = tf.nn.relu(bn)
        result = relu.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 4, 4, 4).astype(np.float32),
        "num_features": 3,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "norm_weight": np.random.rand(3).astype(np.float32),
        "norm_bias": np.random.rand(3).astype(np.float32),
        "freeze_bn": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()