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
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig, default_qconfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    qconfig = input_dict.get("qconfig", default_qconfig)
    inplace = input_dict.get("inplace", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv_bn_relu = nniqat.ConvBnReLU2d(
        input_dict["in_channels"],
        input_dict["out_channels"],
        input_dict["kernel_size"],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        qconfig=qconfig,
        bias=True
    )
    
    conv_bn_relu.weight = nn.Parameter(weight)
    conv_bn_relu.bias = nn.Parameter(bias)
    conv_bn_relu.bn.running_mean = running_mean
    conv_bn_relu.bn.running_var = running_var
    conv_bn_relu.bn.eps = eps
    conv_bn_relu.bn.momentum = momentum
    
    result = conv_bn_relu(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

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
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        qconfig = input_dict.get("qconfig", None)
        inplace = input_dict.get("inplace", False)

        kernel_size = input_dict["kernel_size"]
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        
        strides = [1, stride, stride, 1]
        rates = [1, dilation, dilation, 1]

        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'
        
        weight_reshaped = tf.transpose(weight, perm=[2, 3, 1, 0])

        if groups > 1:
            raise NotImplementedError("Grouped convolutions are not yet supported in this example.")
        
        input_shape = input_dict["input"].shape
        kernel_shape = input_dict["weight"].shape

        if input_shape[1] % groups != 0:
            raise ValueError("The number of input channels must be divisible by the number of groups.")
        if kernel_shape[1] != in_channels:
            raise ValueError(f"Kernel input channels {kernel_shape[1]} do not match input channels {in_channels}")

        conv_out = tf.nn.conv2d(input_tensor, weight_reshaped, strides=strides, padding=padding_tf, dilations=rates)

        gamma = tf.math.sqrt(running_var + eps)

        bn_out = tf.nn.batch_normalization(conv_out, mean=running_mean, variance=running_var, offset=bias, scale=gamma, variance_epsilon=eps)

        relu_out = tf.nn.relu(bn_out)

        result = relu_out.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    in_channels = 3
    out_channels = 16
    input_data = {
        "input": np.random.rand(1, in_channels, 32, 32).astype(np.float32),
        "weight": np.random.rand(out_channels, in_channels, 3, 3).astype(np.float32),
        "bias": np.random.rand(out_channels).astype(np.float32),
        "running_mean": np.random.rand(out_channels).astype(np.float32),
        "running_var": np.random.rand(out_channels).astype(np.float32),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)

    input_data_tf = {k: v for k, v in input_data.items()}
    input_data_tf['input'] = input_data['input'].astype(np.float32)
    input_data_tf['weight'] = input_data['weight'].astype(np.float32)
    input_data_tf['bias'] = input_data['bias'].astype(np.float32)
    input_data_tf['running_mean'] = input_data['running_mean'].astype(np.float32)
    input_data_tf['running_var'] = input_data['running_var'].astype(np.float32)

    tf_result = tensorflow_version(input_data_tf)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()