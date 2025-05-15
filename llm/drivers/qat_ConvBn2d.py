import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig, default_qat_qconfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    gamma = torch.tensor(input_dict["gamma"])
    beta = torch.tensor(input_dict["beta"])

    stride = input_dict.get("stride", (1, 1))
    padding = input_dict.get("padding", (0, 0))
    dilation = input_dict.get("dilation", (1, 1))
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    freeze_bn = input_dict.get("freeze_bn", False)
    qconfig = input_dict.get("qconfig")
    if qconfig is None:
        qconfig = default_qat_qconfig

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        gamma = gamma.cuda()
        beta = beta.cuda()

    conv_bn = nniqat.ConvBn2d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"],
                               stride=stride, padding=padding, dilation=dilation, groups=groups,
                               padding_mode=padding_mode, eps=eps, momentum=momentum, freeze_bn=freeze_bn,
                               qconfig=qconfig)

    conv_bn.weight = nn.Parameter(weight)
    conv_bn.bias = nn.Parameter(bias)
    conv_bn.running_mean = running_mean
    conv_bn.running_var = running_var
    conv_bn.bn.weight = nn.Parameter(gamma)
    conv_bn.bn.bias = nn.Parameter(beta)
    
    result = conv_bn(input_tensor)

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
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        gamma = tf.constant(input_dict["gamma"], dtype=tf.float32)
        beta = tf.constant(input_dict["beta"], dtype=tf.float32)

        stride = input_dict.get("stride", (1, 1))
        padding = input_dict.get("padding", (0, 0))
        dilation = input_dict.get("dilation", (1, 1))
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        freeze_bn = input_dict.get("freeze_bn", False)
        qconfig = input_dict.get("qconfig", None)

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports padding_mode='zeros'")

        input_shape = input_dict["input"].shape
        weight_shape = input_dict["weight"].shape

        if input_shape[3] != weight_shape[1]:
            raise ValueError(f"Input depth ({input_shape[3]}) must match filter depth ({weight_shape[1]})")
        
        stride_list = [1, stride[0], stride[1], 1]
        dilation_list = [1, dilation[0], dilation[1], 1]
        padding_list = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]

        input_tensor_padded = tf.pad(input_tensor, padding_list)

        conv_out = tf.nn.conv2d(input_tensor_padded, weight, strides=stride_list, padding='VALID', dilations=dilation_list)
        conv_out = tf.nn.bias_add(conv_out, bias)

        mean = running_mean
        variance = running_var

        scale = gamma * tf.math.rsqrt(variance + eps)
        offset = beta - mean * scale

        result = scale * conv_out + offset

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "running_mean": np.random.rand(16).astype(np.float32),
        "running_var": np.random.rand(16).astype(np.float32),
        "gamma": np.random.rand(16).astype(np.float32),
        "beta": np.random.rand(16).astype(np.float32),
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "padding_mode": 'zeros',
        "eps": 1e-05,
        "momentum": 0.1,
        "freeze_bn": False,
        "qconfig": None
    }
    try:
        torch_result = torch_version(input_data)
        tf_result = tensorflow_version(input_data)

        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

        print("Success")

    except ValueError as e:
        print(f"ValueError: {e}")
    except AssertionError as e:
        print(f"AssertionError: {e}")


if __name__ == "__main__":
    main()