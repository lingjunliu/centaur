import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig, default_qat_qconfig

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
    conv_bn_relu = nniqat.ConvBnReLU1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig)

    if not cpu:
        conv_bn_relu = conv_bn_relu.cuda()
    
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports 'zeros' padding mode.")

        if groups != 1:
            raise ValueError("TensorFlow does not directly support grouped 1D convolutions. Requires custom implementation.")

        input_tensor_expanded = tf.expand_dims(input_tensor, axis=0)
        input_tensor_expanded = tf.expand_dims(input_tensor_expanded, axis=0)

        kernel_shape = [kernel_size, in_channels, out_channels]
        kernel = tf.random.normal(kernel_shape, dtype=tf.float32)

        if padding > 0:
            input_tensor_expanded = tf.pad(input_tensor_expanded, [[0, 0], [0, 0], [padding, padding], [0, 0]], "CONSTANT")

        conv1d = tf.nn.conv1d(input_tensor_expanded, filters=kernel, stride=stride, padding="VALID")

        if dilation != 1:
            d_format = 'NWC'
            conv1d = tf.nn.dilation_conv2d(input_tensor_expanded, filters=kernel, strides=[1, stride, 1, 1], padding='VALID', dilations=[1, dilation, 1, 1], data_format = d_format)

        if bias:
            bias_tensor = tf.random.normal([out_channels], dtype=tf.float32)
            conv1d = tf.nn.bias_add(conv1d, bias_tensor)

        relu = tf.nn.relu(conv1d)
        
        mean = tf.random.normal([out_channels], dtype=tf.float32)
        variance = tf.random.uniform([out_channels], minval=0.0, maxval=1.0, dtype=tf.float32)
        beta = tf.random.normal([out_channels], dtype=tf.float32)
        gamma = tf.random.normal([out_channels], dtype=tf.float32)

        bn = tf.nn.batch_normalization(relu, mean, variance, beta, gamma, 1e-5)

        result = tf.squeeze(bn, axis=[0,1])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5).astype(np.float32),
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()