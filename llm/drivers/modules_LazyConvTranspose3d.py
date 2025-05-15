import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel = torch.tensor(input_dict["weight"])
    stride = input_dict.get("stride", (1, 1, 1))
    padding = input_dict.get("padding", (0, 0, 0))
    output_padding = input_dict.get("output_padding", (0, 0, 0))
    groups = input_dict.get("groups", 1)
    dilation = input_dict.get("dilation", (1, 1, 1))
    bias = input_dict.get("bias", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        kernel = kernel.cuda()

    lazy_conv_transpose3d = torch.nn.LazyConvTranspose3d(
        input_tensor.shape[1],
        kernel.shape[0],
        kernel.shape[2:],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        bias=bias,
    )

    lazy_conv_transpose3d.weight = torch.nn.Parameter(kernel)
    if 'bias_val' in input_dict:
        lazy_conv_transpose3d.bias = torch.nn.Parameter(torch.tensor(input_dict["bias_val"]))
    elif bias:
        lazy_conv_transpose3d.bias = torch.nn.Parameter(torch.zeros(kernel.shape[0]))

    if not cpu:
        lazy_conv_transpose3d = lazy_conv_transpose3d.cuda()

    result = lazy_conv_transpose3d(input_tensor)

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
        kernel = tf.constant(input_dict["weight"])
        stride = input_dict.get("stride", (1, 1, 1))
        padding = input_dict.get("padding", (0, 0, 0))
        output_padding = input_dict.get("output_padding", (0, 0, 0))
        groups = input_dict.get("groups", 1)
        dilation = input_dict.get("dilation", (1, 1, 1))
        bias = input_dict.get("bias", True)

        input_shape = input_tensor.shape
        kernel_shape = kernel.shape

        if groups != 1:
            raise ValueError("Tensorflow does not support groups != 1 for conv3d_transpose")

        output_shape = (
            input_shape[0],
            kernel_shape[0],
            (input_shape[2] - 1) * stride[0] - 2 * padding[0] + kernel_shape[2] + output_padding[0],
            (input_shape[3] - 1) * stride[1] - 2 * padding[1] + kernel_shape[3] + output_padding[1],
            (input_shape[4] - 1) * stride[2] - 2 * padding[2] + kernel_shape[4] + output_padding[2]
        )

        tf_padding = 'VALID' if padding == (0, 0, 0) else 'SAME'

        result = tf.nn.conv3d_transpose(
            input_tensor,
            kernel,
            output_shape=(input_shape[0], output_shape[2], output_shape[3], output_shape[4], input_shape[1]),
            strides=[1, stride[0], stride[1], stride[2], 1],
            padding=tf_padding,
            data_format="NDHWC",
            dilations=[1, dilation[0], dilation[1], dilation[2], 1]
        )

        if bias:
            if 'bias_val' in input_dict:
                bias_val = tf.constant(input_dict["bias_val"], dtype=tf.float32)
            else:
                bias_val = tf.zeros(kernel_shape[0], dtype=tf.float32)
            result = tf.nn.bias_add(result, bias_val, data_format="NDHWC")

        result = tf.transpose(result, perm=[0, 4, 1, 2, 3])
        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 4, 4, 4).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "output_padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "bias": True,
        "bias_val": np.random.rand(5).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()