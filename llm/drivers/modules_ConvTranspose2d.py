import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = None
    if "bias" in input_dict:
        bias = torch.tensor(input_dict["bias"])
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    dilation = input_dict.get("dilation", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    in_channels = input_tensor.shape[3]
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]

    conv_transpose2d = torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias is not None,
        dilation=dilation,
        padding_mode=padding_mode
    )

    conv_transpose2d.weight.data = weight.permute(0, 3, 1, 2)

    if bias is not None:
        conv_transpose2d.bias.data = bias

    input_tensor = input_tensor.permute(0, 3, 1, 2)

    result = conv_transpose2d(input_tensor)

    if not cpu:
        result = result.cpu()
        
    result = result.permute(0, 2, 3, 1)

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
        bias = None
        if "bias" in input_dict:
            bias = tf.constant(input_dict["bias"])
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        groups = input_dict.get("groups", 1)
        dilation = input_dict.get("dilation", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("Tensorflow only supports padding_mode='zeros'")

        input_shape = input_tensor.shape
        weight_shape = weight.shape
        
        output_shape = [input_shape[0],
                         (input_shape[1] - 1) * stride - 2 * padding + dilation * (weight_shape[2] - 1) + output_padding + 1,
                         (input_shape[2] - 1) * stride - 2 * padding + dilation * (weight_shape[3] - 1) + output_padding + 1,
                         weight_shape[0]]
        
        output_shape_arg = tf.constant(output_shape, dtype=tf.int32)

        weight = tf.transpose(weight, perm=[2, 3, 0, 1])
        result = tf.nn.conv2d_transpose(
            input_tensor,
            weight,
            output_shape=output_shape_arg,
            strides=[1, stride, stride, 1],
            padding="VALID",
            data_format="NHWC"
        )

        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format="NHWC")

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 4, 4, 3).astype(np.float32),
        "weight": np.random.rand(2, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(2).astype(np.float32),
        "stride": 2,
        "padding": 1,
        "output_padding": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()