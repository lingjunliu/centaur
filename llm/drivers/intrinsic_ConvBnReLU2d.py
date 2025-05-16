import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])

    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)
    in_channels = input_tensor.shape[1]
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    conv_bn_relu = torch.nn.intrinsic.ConvBnReLU2d(
        in_channels,
        out_channels,
        kernel_size
    )

    conv_bn_relu.conv.stride = (stride, stride)
    conv_bn_relu.conv.padding = (padding, padding)
    conv_bn_relu.conv.dilation = (dilation, dilation)
    conv_bn_relu.conv.groups = groups
    conv_bn_relu.conv.padding_mode = padding_mode
    conv_bn_relu.bn.eps = eps
    conv_bn_relu.bn.momentum = momentum
    conv_bn_relu.bn.affine = affine
    conv_bn_relu.bn.track_running_stats = track_running_stats

    with torch.no_grad():
        conv_bn_relu.weight.copy_(weight)
        conv_bn_relu.bias.copy_(bias)
        conv_bn_relu.bn.running_mean.copy_(torch.zeros_like(conv_bn_relu.bn.running_mean))
        conv_bn_relu.bn.running_var.copy_(torch.ones_like(conv_bn_relu.bn.running_var))
        if affine:
            conv_bn_relu.bn.weight.copy_(torch.ones_like(conv_bn_relu.bn.weight))
            conv_bn_relu.bn.bias.copy_(torch.zeros_like(conv_bn_relu.bn.bias))

    result = conv_bn_relu(input_tensor)

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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])

        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        if padding_mode == 'zeros':
            padding_tf = 'VALID' if padding == 0 else 'SAME'
        elif padding_mode == 'reflect':
            raise ValueError("Reflect padding not supported in tensorflow")
        elif padding_mode == 'replicate':
            raise ValueError("Replicate padding not supported in tensorflow")
        elif padding_mode == 'circular':
            raise ValueError("Circular padding not supported in tensorflow")
        else:
            raise ValueError("Invalid padding mode")

        if groups > 1:
           raise ValueError("Groups > 1 not supported in tensorflow")

        strides_tf = [1, stride, stride, 1]
        dilations_tf = [1, dilation, dilation, 1]

        conv = tf.nn.conv2d(input_tensor, weight, strides=strides_tf, padding=padding_tf, dilations=dilations_tf)
        conv = tf.nn.bias_add(conv, bias)

        # Calculate moments for batch norm
        axes = [0, 1, 2]
        mean, variance = tf.nn.moments(conv, axes=axes, keepdims=False)

        beta = tf.zeros([weight.shape[0]], dtype=tf.float32)
        gamma = tf.ones([weight.shape[0]], dtype=tf.float32)

        bn = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, eps)

        relu = tf.nn.relu(bn)

        result = relu.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(6, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()