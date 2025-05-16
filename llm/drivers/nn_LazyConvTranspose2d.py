import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    dilation = input_dict.get("dilation", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    device = None
    dtype = None

    if not cpu:
        input_tensor = input_tensor.cuda()

    layer = torch.nn.LazyConvTranspose2d(
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype
    )

    result = layer(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    dilation = input_dict.get("dilation", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    in_channels = input_tensor.shape[1]
    batch_size = input_tensor.shape[0]

    kernel_shape = (kernel_size, kernel_size, out_channels, in_channels // groups)

    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    kernel = tf.Variable(initial_value=kernel_initializer(shape=kernel_shape, dtype='float32'), trainable=True)

    if bias:
        bias_initializer = tf.zeros_initializer()
        bias_val = tf.Variable(initial_value=bias_initializer(shape=(out_channels,), dtype='float32'), trainable=True)
    else:
        bias_val = None

    input_height = input_tensor.shape[2]
    input_width = input_tensor.shape[3]

    output_height = (input_height - 1) * stride + kernel_size - 2 * padding + output_padding
    output_width = (input_width - 1) * stride + kernel_size - 2 * padding + output_padding

    output_shape = [batch_size, output_height, output_width, out_channels]

    paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
    input_padded = tf.pad(tf.transpose(input_tensor, perm=[0, 2, 3, 1]), paddings, "CONSTANT")

    result = tf.nn.conv2d_transpose(
        input_padded,
        kernel,
        output_shape=output_shape,
        strides=[1, stride, stride, 1],
        padding="VALID"
    )

    if output_padding != 0:
        result = result[:, :output_height, :output_width, :]

    if bias:
        result = tf.nn.bias_add(result, bias_val)
    
    result = tf.transpose(result, perm=[0, 3, 1, 2])

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10).astype(np.float32),
        "out_channels": 6,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 1,
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