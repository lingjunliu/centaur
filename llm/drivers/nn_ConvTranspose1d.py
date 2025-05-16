import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    dilation = input_dict.get("dilation", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    conv_transpose1d = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)

    if not cpu:
        conv_transpose1d = conv_transpose1d.cuda()

    result = conv_transpose1d(input_tensor)

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
        output_padding = input_dict.get("output_padding", 0)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        dilation = input_dict.get("dilation", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        input_shape = input_tensor.shape
        kernel_shape = (kernel_size, in_channels // groups, out_channels)

        w = tf.random.normal(kernel_shape)

        output_length = (input_shape[2] - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        output_shape = (input_shape[0], output_length, out_channels)

        output = tf.nn.conv1d_transpose(
            tf.transpose(input_tensor, perm=[0, 2, 1]),
            w,
            output_shape=output_shape,
            strides=stride,
            padding="VALID",
            data_format='NWC',
            dilations=dilation
        )

        if bias:
            b = tf.random.normal([out_channels])
            output = tf.nn.bias_add(output, b, data_format='NWC')

        output = tf.transpose(output, perm=[0, 2, 1])

        result = output.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 5).astype(np.float32),
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 0,
        "groups": 3,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()