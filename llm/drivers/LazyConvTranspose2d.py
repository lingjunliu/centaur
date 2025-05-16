import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict.get("kernel_size")
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    dilation = input_dict.get("dilation", 1)
    out_channels = input_dict.get("out_channels")

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_conv_transpose = torch.nn.LazyConvTranspose2d(
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation
    )

    if not cpu:
        lazy_conv_transpose = lazy_conv_transpose.cuda()
    
    result = lazy_conv_transpose(input_tensor)
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        kernel_size = input_dict.get("kernel_size")
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        dilation = input_dict.get("dilation", 1)
        out_channels = input_dict.get("out_channels")
        in_channels = input_dict.get("in_channels", input_tensor.shape[-1])

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        if len(input_tensor.shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        kernel_shape = (kernel_size[0], kernel_size[1], out_channels, in_channels // groups)
        kernel = tf.random.normal(kernel_shape, dtype=tf.float32)

        output_shape = (input_tensor.shape[0], input_tensor.shape[1] * stride, input_tensor.shape[2] * stride, out_channels)

        result = tf.nn.conv2d_transpose(
            input_tensor,
            kernel,
            output_shape=output_shape,
            strides=(1, stride, stride, 1),
            padding='SAME' if padding > 0 else 'VALID',
            data_format='NHWC',
            dilations=(1, dilation, dilation, 1)
        )

        if isinstance(output_padding, int) and output_padding > 0:
            result = tf.pad(result, [[0, 0], [0, output_padding], [0, output_padding], [0, 0]], "CONSTANT")

        if bias:
            bias_init = tf.random.normal([out_channels], dtype=tf.float32)
            result = tf.nn.bias_add(result, bias_init)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.random.rand(1, 16, 16, 3).astype(np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "out_channels": 5,
        "in_channels": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()