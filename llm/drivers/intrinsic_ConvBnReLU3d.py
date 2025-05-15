import numpy as np
import torch
import torch.nn as nn

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])

    stride = input_dict.get("stride", (1,1,1))
    padding = input_dict.get("padding", (0,0,0))
    dilation = input_dict.get("dilation", (1,1,1))
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    conv3d = nn.Conv3d(
        input_dict["in_channels"],
        input_dict["out_channels"],
        input_dict["kernel_size"],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode
    )
    bn3d = nn.BatchNorm3d(input_dict["out_channels"], eps, momentum, affine, track_running_stats)
    relu = nn.ReLU()
    conv_bn_relu = torch.nn.Sequential(conv3d, bn3d, relu)

    conv3d.weight.data = weight
    conv3d.bias.data = bias
    bn3d.running_mean = torch.tensor(input_dict["running_mean"])
    bn3d.running_var = torch.tensor(input_dict["running_var"])
    if affine:
        bn3d.weight.data = torch.tensor(input_dict["bn_weight"])
        bn3d.bias.data = torch.tensor(input_dict["bn_bias"])

    result = conv_bn_relu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])

        stride = input_dict.get("stride", (1,1,1))
        padding = input_dict.get("padding", (0,0,0))
        dilation = input_dict.get("dilation", (1,1,1))
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports 'zeros' padding mode.")

        if dilation != (1,1,1):
            raise ValueError("TensorFlow does not directly support dilation in conv3d. "
                             "Consider using dilated_convolution3d if needed.")

        if groups != 1:
            raise ValueError("TensorFlow does not directly support groups in conv3d. "
                             "Consider using tf.nn.depthwise_conv3d if needed.")

        strides = [1, stride[0], stride[1], stride[2], 1]
        padding_tf = 'VALID' if padding == (0,0,0) else 'SAME'

        conv = tf.nn.conv3d(input_tensor, weight, strides=strides, padding=padding_tf)
        conv = tf.nn.bias_add(conv, bias)

        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        bn_weight = tf.constant(input_dict["bn_weight"], dtype=tf.float32)
        bn_bias = tf.constant(input_dict["bn_bias"], dtype=tf.float32)

        mean, variance = tf.nn.moments(conv, axes=[0, 1, 2, 3], keepdims=False)

        if track_running_stats:
            bn = tf.nn.batch_normalization(conv, mean, variance, bn_bias, bn_weight, eps)
        else:
            bn = tf.nn.batch_normalization(conv, running_mean, running_var, bn_bias, bn_weight, eps)

        relu = tf.nn.relu(bn)

        result = relu.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": (3,3,3),
        "running_mean": np.random.rand(5).astype(np.float32),
        "running_var": np.random.rand(5).astype(np.float32),
        "bn_weight": np.random.rand(5).astype(np.float32),
        "bn_bias": np.random.rand(5).astype(np.float32),
        "stride": (1,1,1),
        "padding": (0,0,0),
        "dilation": (1,1,1),
        "groups": 1,
        "padding_mode": 'zeros',
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()