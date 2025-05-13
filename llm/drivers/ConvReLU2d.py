import numpy as np
import torch
import torch.nn as nn
import torch.nn.intrinsic.quantized as nniq
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1,))
    padding = input_dict.get("padding", (0,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    conv_relu = nniq.ConvReLU2d(
        input_dict["in_channels"],
        input_dict["out_channels"],
        input_dict["kernel_size"],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        bias=(bias is not None),
    )
    
    conv_relu.weight = nn.Parameter(weight.data)
    if bias is not None:
        conv_relu.bias = nn.Parameter(bias.data)

    result = conv_relu(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        stride = input_dict.get("stride", (1,))
        padding = input_dict.get("padding", (0,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports 'zeros' padding mode.")

        if groups != 1:
            raise ValueError("TensorFlow only supports groups=1 for now.")

        strides = [1, stride[0], stride[0], 1] if isinstance(stride, tuple) else [1, stride, stride, 1]
        rates = [1, dilation[0], dilation[0], 1] if isinstance(dilation, tuple) else [1, dilation, dilation, 1]

        if padding == (0,):
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'

        conv = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_tf, dilations=rates)
        if bias is not None:
            conv = tf.nn.bias_add(conv, bias)
        
        result = tf.nn.relu(conv)
        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()