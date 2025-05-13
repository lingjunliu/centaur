import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.quantized as nnq

    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()

    conv3d = nnq.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)
    
    with torch.no_grad():
        conv3d.weight.data[:] = torch.tensor(input_dict["weight"])
        conv3d.bias.data[:] = torch.tensor(input_dict["bias"])

    if not cpu:
        conv3d = conv3d.cuda()

    result = conv3d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

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
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'VALID')

        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(padding, int):
            if padding == 0:
                padding_mode = 'VALID'
            else:
                padding_mode = 'SAME'
        else:
            padding_mode = 'VALID'

        if padding_mode == 'SAME':
            result = tf.nn.conv3d(input_tensor[tf.newaxis, ...], weight, strides=(1, stride[0], stride[1], stride[2], 1), padding=padding_mode) + bias
        else:
            result = tf.nn.conv3d(input_tensor[tf.newaxis, ...], weight, strides=(1, stride[0], stride[1], stride[2], 1), padding=padding_mode) + bias

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 2, 3, 4, 5).astype(np.float32),
        "in_channels": 2,
        "out_channels": 3,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "weight": np.random.rand(3, 2, 2, 2, 2).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"][0], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()