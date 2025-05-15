import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

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

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode
    )
    
    if not cpu:
        conv = conv.cuda()

    result = conv(input_tensor)
    result = torch.relu(result)

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
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports 'zeros' padding mode.")

        if len(input_tensor.shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])

        kernel_shape = (kernel_size[0], kernel_size[1], in_channels // groups, out_channels)
        kernel = tf.random.normal(shape=kernel_shape)

        strides = [1, stride[0], stride[1], 1]
        rates = [1, dilation, dilation, 1]

        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'

        if groups > 1:
             kernel_shape = (kernel_size[0], kernel_size[1], in_channels // groups, out_channels)
             kernel = tf.random.normal(shape=kernel_shape)
             
        conv = tf.nn.depthwise_conv2d(input_tensor, kernel, strides=strides, padding=padding_tf, dilations=rates)
        if bias:
            bias_val = tf.random.normal(shape=(out_channels,))
            conv = tf.nn.bias_add(conv, bias_val)
        
        result = tf.nn.relu(conv)
        result = tf.transpose(result, perm=[0, 3, 1, 2])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()