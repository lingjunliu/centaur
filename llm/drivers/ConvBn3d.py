import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])

    stride = input_dict.get("stride", (1, 1, 1))
    padding = input_dict.get("padding", (0, 0, 0))
    dilation = input_dict.get("dilation", (1, 1, 1))
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    freeze_bn = input_dict.get("freeze_bn", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    qconfig = QConfig(
        activation=torch.quantization.default_observer,
        weight=torch.quantization.default_weight_observer
    )

    module = nniqat.ConvBn3d(
        input_dict["in_channels"],
        input_dict["out_channels"],
        input_dict["kernel_size"],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        eps=eps,
        momentum=momentum,
        freeze_bn=freeze_bn,
        qconfig=qconfig,
    )

    module.weight = nn.Parameter(weight)
    module.bias = nn.Parameter(bias)
    module.running_mean = running_mean
    module.running_var = running_var

    module.eval()
    with torch.no_grad():
        result = module(input_tensor)

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

        stride = input_dict.get("stride", (1, 1, 1))
        padding = input_dict.get("padding", (0, 0, 0))
        dilation = input_dict.get("dilation", (1, 1, 1))
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        freeze_bn = input_dict.get("freeze_bn", False)

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports padding_mode='zeros'")

        if groups != 1:
            raise ValueError("TensorFlow does not directly support groups > 1 for Conv3D")

        strides = [1] + list(stride) + [1]
        dilations = [1] + list(dilation) + [1]
        padding_tf = "VALID" if padding == (0, 0, 0) else "SAME"
        
        weight_reshaped = tf.transpose(weight, perm=[2, 3, 4, 1, 0])
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])

        conv_out = tf.nn.conv3d(input_tensor, weight_reshaped, strides=strides, padding=padding_tf, dilations=dilations, data_format="NDHWC")
        conv_out = tf.nn.bias_add(conv_out, bias, data_format="NDHWC")

        # Calculate batch norm manually
        mean = running_mean
        variance = running_var
        
        scale = tf.math.rsqrt(variance + eps)
        offset = -mean * scale
        
        result = (conv_out - offset) / scale

        result = tf.transpose(result, perm=[0, 4, 1, 2, 3]).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_shape = (1, 3, 10, 10, 10)
    in_channels = 3
    out_channels = 5
    kernel_size = (3, 3, 3)

    input_data = {
        "input": np.random.rand(*input_shape).astype(np.float32),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "groups": 1,
        "padding_mode": 'zeros',
        "eps": 1e-05,
        "momentum": 0.1,
        "freeze_bn": False,
        "weight": np.random.rand(out_channels, in_channels, *kernel_size).astype(np.float32),
        "bias": np.random.rand(out_channels).astype(np.float32),
        "running_mean": np.random.rand(out_channels).astype(np.float32),
        "running_var": np.random.rand(out_channels).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()