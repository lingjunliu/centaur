import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    input_tensor = input_tensor.permute(0, 3, 1, 2)

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_conv = nn.LazyConv2d(input_dict.get("out_channels"), input_dict.get("kernel_size"), stride=input_dict.get("stride", 1), padding=input_dict.get("padding", 0), dilation=input_dict.get("dilation", 1), groups=input_dict.get("groups", 1), bias=input_dict.get("bias", True))

    result = lazy_conv(input_tensor)

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
        out_channels = input_dict.get("out_channels")
        kernel_size = input_dict.get("kernel_size")
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)

        input_shape = input_tensor.shape
        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        elif len(input_shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        in_channels = int(input_tensor.shape[-1])

        if groups > 1:
            kernel_shape = [kernel_size, kernel_size, in_channels // groups, out_channels // groups]
        else:
            kernel_shape = [kernel_size, kernel_size, in_channels, out_channels]

        kernel = tf.Variable(tf.random.normal(kernel_shape))

        if padding > 0:
            pad_size = padding
            input_tensor = tf.pad(input_tensor, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")

        if groups > 1:
            result = tf.nn.convolution(input_tensor, kernel, padding="VALID", strides=[stride], dilation_rate=[dilation])
        else:
            result = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride, stride, 1], padding="VALID", dilations=[1, dilation, dilation, 1])


        if bias:
            bias_val = tf.Variable(tf.zeros([out_channels]))
            result = tf.nn.bias_add(result, bias_val)

        result = tf.transpose(result, perm=[0, 3, 1, 2]).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 32, 32, 3).astype(np.float32),
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]

    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()