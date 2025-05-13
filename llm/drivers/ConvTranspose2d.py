import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

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

    conv_transpose2d = torch.nn.ConvTranspose2d(
        in_channels=input_dict["in_channels"],
        out_channels=input_dict["out_channels"],
        kernel_size=input_dict["kernel_size"],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=True if bias is not None else False,
        dilation=dilation,
    )
    
    conv_transpose2d.weight.data = weight
    if bias is not None:
        conv_transpose2d.bias.data = bias

    result = conv_transpose2d(input_tensor.permute(0, 3, 1, 2))

    if not cpu:
        result = result.cpu()

    return {"result": result.permute(0, 2, 3, 1).detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

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

        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]

        output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        output_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

        output_shape = (batch_size, output_height, output_width, out_channels)

        weight_reshaped = tf.transpose(weight, perm=[2, 3, 1, 0])

        if groups > 1:
            raise NotImplementedError("Groups > 1 is not implemented for tensorflow version.")

        result = tf.nn.conv2d_transpose(
            input=input_tensor,
            filters=weight_reshaped,
            output_shape=output_shape,
            strides=[1, stride[0], stride[1], 1],
            padding="VALID",
            data_format="NHWC",
        )

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 5, 5, 3).astype(np.float32),
        "weight": np.random.rand(3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "in_channels": 3,
        "out_channels": 3,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "dilation": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()