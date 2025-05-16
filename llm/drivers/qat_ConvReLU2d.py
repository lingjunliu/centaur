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
    from torch.ao.quantization import QConfig, default_qat_qconfig

    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()

    qconfig = default_qat_qconfig
    conv_relu = nniqat.ConvReLU2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig)
    
    conv_relu.weight = nn.Parameter(torch.tensor(input_dict["weight"]))
    if bias:
      conv_relu.bias = nn.Parameter(torch.tensor(input_dict["bias_data"]))

    if not cpu:
        conv_relu = conv_relu.cuda()
        conv_relu.weight = nn.Parameter(conv_relu.weight.cuda())
        if bias:
          conv_relu.bias = nn.Parameter(conv_relu.bias.cuda())

    result = conv_relu(input_tensor)

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
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        weight = tf.constant(input_dict["weight"])

        if padding_mode != 'zeros':
            raise ValueError("Tensorflow only supports 'zeros' padding mode.")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)
        
        input_shape = input_tensor.shape
        weight_shape = weight.shape

        if input_shape[3] % groups != 0:
            raise ValueError(f"The number of input channels ({input_shape[3]}) must be evenly divisible by the number of groups ({groups}).")

        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")

        if weight_shape[1] != in_channels // groups:
            raise ValueError(f"The number of weight input channels ({weight_shape[1]}) must match the number of input channels // groups ({in_channels // groups}).")
            
        strides = [1, stride[0], stride[1], 1]
        rates = [1, dilation, dilation, 1]
        padding_tf = 'VALID' if padding == (0, 0) else 'SAME'

        weight_reshaped = tf.transpose(weight, perm=[2, 3, 1, 0])
        
        result_conv = tf.nn.conv2d(input_tensor, weight_reshaped, strides=strides, padding=padding_tf, dilations=rates)
        
        if bias:
            bias_data = tf.constant(input_dict["bias_data"])
            result_conv = tf.nn.bias_add(result_conv, bias_data)

        result = tf.nn.relu(result_conv)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros',
        "weight": np.random.rand(5, 3, 3, 3).astype(np.float32),
        "bias_data": np.random.rand(5).astype(np.float32)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()