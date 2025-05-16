import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"], requires_grad=False)
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

    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    if not cpu:
        conv = conv.cuda()

    result = conv(input_tensor)

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
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        input_shape = input_dict["input"].shape
        
        if in_channels % groups != 0:
            raise ValueError("The value of in_channels must be divisible by groups.")


        kernel_shape = (kernel_size[0], kernel_size[1], in_channels // groups, out_channels)
        kernel_initializer = tf.keras.initializers.GlorotUniform()
        kernel = tf.Variable(initial_value=kernel_initializer(shape=kernel_shape, dtype='float32'), trainable=True)

        if bias:
            bias_initializer = tf.zeros_initializer()
            bias_shape = (out_channels,)
            bias_var = tf.Variable(initial_value=bias_initializer(shape=bias_shape, dtype='float32'), trainable=True)
        else:
            bias_var = None
        
        if padding_mode == 'zeros':
            padding_tf = 'VALID' if padding == (0, 0) else 'SAME'
        elif padding_mode == 'reflect':
            input_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], "REFLECT")
            padding_tf = 'VALID'
        elif padding_mode == 'replicate':
            input_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], "SYMMETRIC")
            padding_tf = 'VALID'
        elif padding_mode == 'circular':
            input_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], "CONSTANT")
            padding_tf = 'VALID'
        else:
            raise ValueError("Invalid padding mode")
        
        result = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride[0], stride[1], 1], padding=padding_tf, dilations=[1, dilation[0], dilation[1], 1])

        if bias_var is not None:
            result = tf.nn.bias_add(result, bias_var)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()