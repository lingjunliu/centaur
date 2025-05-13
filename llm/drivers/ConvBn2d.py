import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig
    from torch.quantization import default_observer

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    freeze_bn = input_dict.get("freeze_bn", False)
    if "qconfig" not in input_dict or input_dict["qconfig"] is None:
        qconfig = QConfig(activation=default_observer.MovingAveragePerChannelObserver, weight=default_observer.default_weight_observer)
    else:
        qconfig = input_dict["qconfig"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        
    module = nniqat.ConvBn2d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"], stride=input_dict["stride"], padding=input_dict["padding"], dilation=input_dict["dilation"], groups=input_dict["groups"], bias=True, eps=eps, momentum=momentum, qconfig=qconfig)
    
    module.weight = nn.Parameter(weight)
    module.bias = nn.Parameter(bias)
    module.bn.running_mean = running_mean
    module.bn.running_var = running_var
    module.bn.eps = eps
    module.bn.momentum = momentum
    module.freeze_bn = freeze_bn

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
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        freeze_bn = input_dict.get("freeze_bn", False)
        qconfig = input_dict.get("qconfig", None)

        kernel_size = input_dict["kernel_size"]
        stride = input_dict["stride"]
        padding = input_dict["padding"]
        dilation = input_dict["dilation"]
        groups = input_dict["groups"]
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        
        weight_reshaped = tf.transpose(weight, perm=[2, 3, 1, 0])
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)
            
        if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        elif isinstance(padding, tuple):
            if isinstance(padding[0], int):
                padding = ((padding[0], padding[0]), (padding[1], padding[1]))
            
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        padding_config = 'VALID'
        if padding[0][0] > 0 or padding[0][1] > 0 or padding[1][0] > 0 or padding[1][1] > 0:
            input_padded = tf.pad(input_tensor, [[0,0], [padding[0][0], padding[0][1]], [padding[1][0], padding[1][1]], [0,0]])
            padding_config = 'VALID'
        else:
            input_padded = input_tensor
        
        conv_result = tf.nn.conv2d(input_padded, weight_reshaped, strides=[1, stride[0], stride[1], 1], padding=padding_config, dilations=[1, dilation[0], dilation[1], 1])
        conv_result = tf.nn.bias_add(conv_result, bias)

        scale = tf.math.rsqrt(running_var + eps)
        bn_weight = weight * scale
        bn_bias = bias - running_mean * scale
    
        result = tf.nn.batch_normalization(conv_result, mean=running_mean, variance=running_var, offset=bn_bias, scale=bn_weight, variance_epsilon=eps)

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
        "eps": 1e-5,
        "momentum": 0.1,
        "freeze_bn": False,
        "qconfig": None,
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()