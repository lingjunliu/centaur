import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    input_tensor = torch.tensor(input_dict["input"])
    kernel = torch.tensor(input_dict["weight"])
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    if "bias_tensor" in input_dict:
        bias_tensor = torch.tensor(input_dict["bias_tensor"])
    else:
        bias_tensor = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        kernel = kernel.cuda()
        if bias_tensor is not None:
            bias_tensor = bias_tensor.cuda()

    if bias:
        if bias_tensor is None:
            layer = torch.nn.LazyConvTranspose1d(input_tensor.shape[1] * groups)
            layer.weight = torch.nn.Parameter(kernel)
            result = layer(input_tensor)
        else:
            layer = torch.nn.ConvTranspose1d(kernel.shape[1], kernel.shape[0], kernel_size=kernel.shape[2], stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=True)
            layer.weight = torch.nn.Parameter(kernel)
            layer.bias = torch.nn.Parameter(bias_tensor)
            result = layer(input_tensor)
    else:
        layer = torch.nn.ConvTranspose1d(kernel.shape[1], kernel.shape[0], kernel_size=kernel.shape[2], stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=False)
        layer.weight = torch.nn.Parameter(kernel)
        result = layer(input_tensor)

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
        kernel = tf.constant(input_dict["weight"])
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)

        input_shape = input_tensor.shape
        kernel_shape = kernel.shape

        if groups != 1:
            raise NotImplementedError("groups != 1 not implemented for tensorflow")

        if isinstance(stride, int):
            stride = [stride]
        if isinstance(dilation, int):
            dilation = [dilation]

        strides = [1] + stride + [1]
        dilations = [1] + dilation + [1]

        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'

        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        kernel = tf.cast(kernel, dtype=tf.float32)
        
        kernel_transposed = tf.transpose(kernel, perm=[2, 1, 0])
        kernel_expanded = tf.expand_dims(kernel_transposed, axis=0)

        result = tf.nn.conv2d_transpose(
            tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=0),
            kernel_expanded,
            output_shape=(1, 1, (input_shape[2] - 1) * strides[1] + kernel_shape[2] - 2 * padding + output_padding, kernel_shape[1]),
            strides=[1, 1, strides[1], 1],
            padding=padding_tf,
            dilations=[1, 1, dilations[1], 1]
        )
        result = tf.squeeze(result, axis=[0,1])

        if "bias_tensor" in input_dict and bias:
            bias_tensor = tf.constant(input_dict["bias_tensor"], dtype=tf.float32)
            result = tf.add(result, bias_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10).astype(np.float32),
        "weight": np.random.rand(4, 3, 2).astype(np.float32),
        "bias_tensor": np.random.rand(4).astype(np.float32),
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "dilation": 1,
        "groups": 1,
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()