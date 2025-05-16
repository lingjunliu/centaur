import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.intrinsic.qat.modules.conv_fused import ConvBnReLU2d, ConvReLU2d, ConvBn2d
    from torch.quantization import QConfig
    from torch.quantization.observer import default_observer, default_weight_observer

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    bn_weight = torch.tensor(input_dict["bn_weight"]) if "bn_weight" in input_dict else None
    bn_bias = torch.tensor(input_dict["bn_bias"]) if "bn_bias" in input_dict else None
    bn_running_mean = torch.tensor(input_dict["bn_running_mean"]) if "bn_running_mean" in input_dict else None
    bn_running_var = torch.tensor(input_dict["bn_running_var"]) if "bn_running_var" in input_dict else None
    eps = input_dict.get("eps", 1e-05)
    relu = input_dict.get("relu", False)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
        if bn_weight is not None:
            bn_weight = bn_weight.cuda()
        if bn_bias is not None:
            bn_bias = bn_bias.cuda()
        if bn_running_mean is not None:
            bn_running_mean = bn_running_mean.cuda()
        if bn_running_var is not None:
            bn_running_var = bn_running_var.cuda()

    qconfig = QConfig(activation=default_observer, weight=default_weight_observer)

    if relu:
        if bn_weight is not None:
            conv = ConvBnReLU2d(input_tensor.shape[1], weight.shape[0], weight.shape[2], stride=stride, padding=padding, dilation=dilation, groups=groups, qconfig=qconfig)
            conv.weight = torch.nn.Parameter(weight)
            conv.bias = torch.nn.Parameter(bias) if bias is not None else None
            conv.bn.weight = torch.nn.Parameter(bn_weight)
            conv.bn.bias = torch.nn.Parameter(bn_bias)
            conv.bn.running_mean = bn_running_mean
            conv.bn.running_var = bn_running_var
            conv.bn.eps = eps

        else:
             conv = ConvReLU2d(input_tensor.shape[1], weight.shape[0], weight.shape[2], stride=stride, padding=padding, dilation=dilation, groups=groups, qconfig=qconfig)
             conv.weight = torch.nn.Parameter(weight)
             conv.bias = torch.nn.Parameter(bias) if bias is not None else None
    else:
        conv = ConvBn2d(input_tensor.shape[1], weight.shape[0], weight.shape[2], stride=stride, padding=padding, dilation=dilation, groups=groups, qconfig=qconfig)
        conv.weight = torch.nn.Parameter(weight)
        conv.bias = torch.nn.Parameter(bias) if bias is not None else None
        conv.bn.weight = torch.nn.Parameter(bn_weight)
        conv.bn.bias = torch.nn.Parameter(bn_bias)
        conv.bn.running_mean = bn_running_mean
        conv.bn.running_var = bn_running_var
        conv.bn.eps = eps
    result = conv(input_tensor)

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
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        bn_weight = tf.constant(input_dict["bn_weight"]) if "bn_weight" in input_dict else None
        bn_bias = tf.constant(input_dict["bn_bias"]) if "bn_bias" in input_dict else None
        bn_running_mean = tf.constant(input_dict["bn_running_mean"]) if "bn_running_mean" in input_dict else None
        bn_running_var = tf.constant(input_dict["bn_running_var"]) if "bn_running_var" in input_dict else None
        eps = input_dict.get("eps", 1e-05)
        relu = input_dict.get("relu", False)
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        if isinstance(stride, int):
            stride = [1, stride, stride, 1]
        
        if isinstance(dilation, int):
            dilation = [dilation, dilation]

        if padding == 0:
            padding = "VALID"
        else:
            padding = "SAME"

        input_shape = input_tensor.shape
        weight_shape = weight.shape

        in_channels = input_shape[-1]
        kernel_channels = weight_shape[1]

        if in_channels % groups != 0 or kernel_channels != in_channels // groups:
            weight = tf.reshape(weight, (weight_shape[0], in_channels, weight_shape[2], weight_shape[3]))
            groups = 1

        if groups == 1:
            conv = tf.nn.conv2d(input_tensor, weight, strides=stride, padding=padding, dilations=[1, dilation[0], dilation[1], 1])
        else:
            input_channels = input_tensor.shape[-1]
            kernel_size = weight.shape[-1]
            if input_channels % groups != 0 or kernel_size % groups != 0:
                raise ValueError('Input channels and kernel size must be divisible by the number of groups.')

            channel_axis = 3 if len(input_tensor.shape) == 4 else 1
            channel_depth = input_channels // groups

            input_list = tf.split(input_tensor, num_or_size_splits=groups, axis=channel_axis)
            weight_list = tf.split(weight, num_or_size_splits=groups, axis=3)
            output_list = [tf.nn.conv2d(i, k, strides=stride, padding=padding, dilations=[1, dilation[0], dilation[1], 1]) for i, k in zip(input_list, weight_list)]
            conv = tf.concat(output_list, axis=channel_axis)


        if bias is not None:
            conv = tf.nn.bias_add(conv, bias)

        if bn_weight is not None:

            mean, variance = tf.nn.moments(conv, axes=[0, 1, 2])
            conv = tf.nn.batch_normalization(conv, mean, variance, bn_bias, bn_weight, eps)
        

        if relu:
            conv = tf.nn.relu(conv)
        
        result = conv.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(6, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "bn_weight": np.random.rand(6).astype(np.float32),
        "bn_bias": np.random.rand(6).astype(np.float32),
        "bn_running_mean": np.random.rand(6).astype(np.float32),
        "bn_running_var": np.random.rand(6).astype(np.float32),
    }

    torch_result = torch_version(input_data)

    input_data_tf = {k:v for k,v in input_data.items()}

    tf_result = tensorflow_version(input_data_tf)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()