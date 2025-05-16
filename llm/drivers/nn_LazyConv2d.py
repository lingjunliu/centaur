import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    device = None
    dtype = None

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_conv2d = nn.LazyConv2d(out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

    if not cpu:
        lazy_conv2d = lazy_conv2d.cuda()

    result = lazy_conv2d(input_tensor)

    if not cpu:
        result = result.cpu()
        
    result = result.detach().numpy()
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        in_channels = input_tensor.shape[3]
        
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        kernel_shape = (kernel_size, kernel_size, in_channels // groups, out_channels)
        kernel = tf.Variable(tf.random.normal(kernel_shape))
        if bias:
            bias_val = tf.Variable(tf.random.normal((out_channels,)))

        padding_enum = {'zeros': 'VALID', 'reflect': 'REFLECT', 'replicate': 'CONSTANT', 'circular': 'CONSTANT'}
        
        if padding_mode not in padding_enum:
            raise ValueError(f"Invalid padding mode: {padding_mode}")
        
        if padding_mode == 'CONSTANT':
            padding_type = 'VALID'
        else:
            padding_type = padding_enum.get(padding_mode)
        
        if padding_type == 'REFLECT':
             paddings = [[0,0], [padding, padding], [padding, padding], [0,0]]
             input_padded = tf.pad(input_tensor, paddings, mode='REFLECT')
             padding_type = 'VALID'
        elif padding_type == 'CONSTANT':
            paddings = [[0,0], [padding, padding], [padding, padding], [0,0]]
            input_padded = tf.pad(input_tensor, paddings, mode='CONSTANT')
            padding_type = 'VALID'
        else:
            input_padded = input_tensor
        
        result = tf.nn.conv2d(
            input_padded,
            kernel,
            strides=[1, stride, stride, 1],
            padding=padding_type,
            dilations=[1, dilation, dilation, 1],
        )

        if bias:
            result = tf.nn.bias_add(result, bias_val)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "out_channels": 4,
        "kernel_size": 3,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    
    input_data_tf = {k: v for k, v in input_data.items()}
    input_data_tf["input"] = np.transpose(input_data["input"], (0, 2, 3, 1))

    tf_result = tensorflow_version(input_data_tf)
    tf_result["result"] = np.transpose(tf_result["result"], (0, 3, 1, 2))
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()