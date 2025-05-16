import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel = torch.tensor(input_dict["kernel"])
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        kernel = kernel.cuda()

    lazy_conv_transpose1d = torch.nn.LazyConvTranspose1d(
        out_channels=kernel.shape[1],
        kernel_size=kernel.shape[2],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    lazy_conv_transpose1d.weight.data = kernel
    

    if bias:
        lazy_conv_transpose1d.bias.data = torch.tensor(input_dict["bias_val"])
        if not cpu:
            lazy_conv_transpose1d.bias.data = lazy_conv_transpose1d.bias.data.cuda()
    else:
        lazy_conv_transpose1d.bias = None

    if not cpu:
        lazy_conv_transpose1d = lazy_conv_transpose1d.cuda()
    result = lazy_conv_transpose1d(input_tensor)

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
        kernel = tf.constant(input_dict["kernel"])
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)

        input_shape = input_tensor.shape.as_list()
        kernel_shape = kernel.shape.as_list()

        if padding == 'valid':
            padding_tf = 'VALID'
        elif padding == 'same':
            padding_tf = 'SAME'
        else:
            padding_tf = 'VALID'

        # Transpose kernel to match expected format [depth, input_depth, output_depth]
        kernel = tf.transpose(kernel, perm=[2, 0, 1])

        output = tf.nn.conv1d_transpose(
            input_tensor,
            kernel,
            output_shape=[input_shape[0], (input_shape[2] - 1) * stride + kernel_shape[0] - 2 * padding + output_padding, kernel_shape[2]],
            strides=stride,
            padding=padding_tf,
            data_format='NWC',
            dilations=dilation
        )
        
        if bias:
            bias_val = tf.constant(input_dict["bias_val"], dtype=tf.float32)
            output = tf.nn.bias_add(output, bias_val, data_format='NWC')

        result = output.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10).astype(np.float32),
        "kernel": np.random.rand(3, 3, 4).astype(np.float32),
        "stride": 2,
        "padding": 1,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "bias_val": np.random.rand(4).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()