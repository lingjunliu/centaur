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
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    input_tensor = input_tensor.permute(0, 3, 1, 2)

    m = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    
    if not cpu:
        m = m.cuda()

    result = m(input_tensor)

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
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(padding, int):
            padding = (padding, padding)

        input_shape = input_tensor.shape
        
        kernel_shape = (kernel_size[0], kernel_size[1], in_channels // groups, out_channels)

        kernel = tf.random.normal(kernel_shape)
        
        if bias:
            bias_val = tf.random.normal((out_channels,))
        else:
            bias_val = None

        if padding_mode == 'zeros':
            if padding == 0:
                padding_tf = 'VALID'
            else:
                padding_tf = 'SAME'
        elif padding_mode == 'reflect' or padding_mode == 'replicate' or padding_mode == 'circular':
            raise NotImplementedError("Padding mode not supported in tensorflow")
        else:
            raise ValueError("Invalid padding mode")

        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        result = tf.nn.conv2d(
            input=input_tensor,
            filters=kernel,
            strides=(1, stride[0], stride[1], 1),
            padding=padding_tf,
            dilations=(1, dilation[0], dilation[1], 1),
            data_format='NHWC'
        )
        
        if bias:
            result = tf.nn.bias_add(result, bias_val, data_format='NHWC')

        result = tf.transpose(result, perm=[0, 3, 1, 2]).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 32, 32, 3).astype(np.float32),
        "in_channels": 3,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()