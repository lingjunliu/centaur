import numpy as np
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

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

    m = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
    
    if not cpu:
        m = m.cuda()
    
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
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

        if isinstance(kernel_size, int):
            kernel_size_height = kernel_size
            kernel_size_width = kernel_size
        else:
            kernel_size_height = kernel_size[0]
            kernel_size_width = kernel_size[1]

        if isinstance(stride, int):
            stride_height = stride
            stride_width = stride
        else:
            stride_height = stride[0]
            stride_width = stride[1]

        if isinstance(padding, int):
            padding_height = padding
            padding_width = padding
        else:
            padding_height = padding[0]
            padding_width = padding[1]

        if isinstance(dilation, int):
            dilation_height = dilation
            dilation_width = dilation
        else:
            dilation_height = dilation[0]
            dilation_width = dilation[1]

        if isinstance(output_padding, int):
            output_padding_height = output_padding
            output_padding_width = output_padding
        else:
            output_padding_height = output_padding[0]
            output_padding_width = output_padding[1]

        if len(input_shape) == 4:
            N, H_in, W_in, C_in = input_shape
        elif len(input_shape) == 3:
            C_in, H_in, W_in = input_shape
            N = 1
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        else:
            raise ValueError("Input tensor must be 3D or 4D")
        
        H_out = (H_in - 1) * stride_height - 2 * padding_height + dilation_height * (kernel_size_height - 1) + output_padding_height + 1
        W_out = (W_in - 1) * stride_width - 2 * padding_width + dilation_width * (kernel_size_width - 1) + output_padding_width + 1

        output_shape = (N, H_out, W_out, out_channels)

        weight_shape = (kernel_size_height, kernel_size_width, C_in, out_channels)
        
        stddev = np.sqrt(groups / (out_channels * kernel_size_height * kernel_size_width))

        weight_values = tf.random.truncated_normal(shape=weight_shape, mean=0.0, stddev=stddev, dtype=tf.float32)
        weights = tf.Variable(weight_values)

        if bias:
            bias_shape = (out_channels,)
            bias_values = tf.random.truncated_normal(shape=bias_shape, mean=0.0, stddev=stddev, dtype=tf.float32)
            bias_tf = tf.Variable(bias_values)
        else:
            bias_tf = None

        strides = [1, stride_height, stride_width, 1]
        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'

        result = tf.nn.conv2d_transpose(input_tensor, weights, output_shape=output_shape, strides=strides, padding=padding_tf)

        if bias_tf is not None:
            result = tf.nn.bias_add(result, bias_tf)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()