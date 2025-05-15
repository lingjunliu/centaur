import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    weight_bn = torch.tensor(input_dict["weight_bn"])
    bias_bn = torch.tensor(input_dict["bias_bn"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight_bn = weight_bn.cuda()
        bias_bn = bias_bn.cuda()

    with torch.no_grad():
        conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        bn3d = nn.BatchNorm3d(out_channels, eps, momentum, affine, track_running_stats)
        conv3d.weight.data = weight
        bn3d.weight.data = weight_bn
        bn3d.bias.data = bias_bn
        bn3d.running_mean.data = running_mean
        bn3d.running_var.data = running_var
        
        result = conv3d(input_tensor)
        result = bn3d(result)
        if affine:
            result = result + bias.reshape(1, -1, 1, 1, 1)

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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        weight_bn = tf.constant(input_dict["weight_bn"])
        bias_bn = tf.constant(input_dict["bias_bn"])

        # Convolution
        input_shape = input_tensor.shape
        if padding > 0:
            input_tensor = tf.pad(input_tensor, [[0,0],[padding,padding],[padding,padding],[padding,padding],[0,0]])
        
        stride_list = [1, stride, stride, stride, 1]
        dilation_list = [1, dilation, dilation, dilation, 1]
        
        weight_transposed = tf.transpose(weight, perm=[2, 3, 4, 1, 0])  # k, k, k, in_channels, out_channels

        conv_result = tf.nn.conv3d(input_tensor, weight_transposed, strides=stride_list, padding='VALID', dilations=dilation_list)

        # Batch Normalization
        inv = tf.math.rsqrt(running_var + eps)
        normalized = (conv_result - running_mean) * inv
        bn_result = weight_bn * normalized + bias_bn

        if affine:
            bn_result = bn_result + bias
        result = bn_result

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 3,
        "weight": np.random.rand(3, 3, 3, 3, 5).astype(np.float32), # Corrected shape for pytorch
        "bias": np.random.rand(5).astype(np.float32),
        "running_mean": np.random.rand(5).astype(np.float32),
        "running_var": np.random.rand(5).astype(np.float32),
        "weight_bn": np.random.rand(5).astype(np.float32),
        "bias_bn": np.random.rand(5).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()