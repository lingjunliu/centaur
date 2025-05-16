import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    dilation = input_dict.get("dilation", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    in_channels = input_tensor.shape[1]

    conv_transpose3d = torch.nn.ConvTranspose3d(
        in_channels=in_channels,
        out_channels=weight.shape[0],
        kernel_size=weight.shape[2:],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias is not None,
        dilation=dilation
    )

    conv_transpose3d.weight.data = weight

    if bias is not None:
        conv_transpose3d.bias.data = bias

    result = conv_transpose3d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

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
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        groups = input_dict.get("groups", 1)
        dilation = input_dict.get("dilation", 1)

        input_shape = tf.shape(input_tensor)
        weight_shape = tf.shape(weight)

        if isinstance(stride, int):
            stride = [stride, stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        if isinstance(dilation, int):
            dilation = [dilation, dilation, dilation]

        output_shape = [
            input_shape[0],
            weight_shape[0],
            (input_shape[2] - 1) * stride[0] - 2 * padding[0] + weight_shape[2] + output_padding,
            (input_shape[3] - 1) * stride[1] - 2 * padding[1] + weight_shape[3] + output_padding,
            (input_shape[4] - 1) * stride[2] - 2 * padding[2] + weight_shape[4] + output_padding
        ]

        strides = [1, stride[0], stride[1], stride[2], 1]

        if isinstance(padding, int):
            padding_type = 'VALID' if padding == 0 else 'SAME'
        else:
            padding_type = 'VALID'

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])
        weight = tf.transpose(weight, perm=[2, 3, 4, 1, 0])

        output_shape_tensor = tf.stack(output_shape)

        result = tf.nn.conv3d_transpose(
            input_tensor,
            weight,
            output_shape=output_shape_tensor,
            strides=strides,
            padding=padding_type
        )

        result = tf.transpose(result, perm=[0, 4, 1, 2, 3])

        if bias is not None:
            bias = tf.constant(bias)
            result = tf.add(result, bias)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 4, 4, 4).astype(np.float32),
        "weight": np.random.rand(2, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(2).astype(np.float32),
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 1,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()