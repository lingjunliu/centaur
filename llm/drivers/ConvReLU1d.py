import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.quantized as nniq

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

    conv_relu = nniq.ConvReLU1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    
    if not cpu:
        conv_relu = conv_relu.cuda()

    with torch.no_grad():
        conv_relu.weight.data.copy_(torch.tensor(input_dict["weight"]))
        if bias:
            conv_relu.bias.data.copy_(torch.tensor(input_dict["bias_val"]))
        
        if not cpu:
            conv_relu.weight.data.copy_(conv_relu.weight.data.cuda())
            if bias:
                conv_relu.bias.data.copy_(conv_relu.bias.data.cuda())

    result = conv_relu(input_tensor)

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
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        if bias:
            bias_val = tf.constant(input_dict["bias_val"], dtype=tf.float32)

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        weight = tf.expand_dims(weight, axis=1)
        
        if padding_mode == 'zeros':
            padding_mode = 'VALID'
            if padding != 0:
                paddings = [[0, 0], [0, 0], [padding, padding], [0, 0]]
                input_tensor = tf.pad(input_tensor, paddings, "CONSTANT")
        elif padding_mode == 'reflect':
            raise ValueError("Tensorflow does not support reflect padding")
        elif padding_mode == 'replicate':
            raise ValueError("Tensorflow does not support replicate padding")
        elif padding_mode == 'circular':
            raise ValueError("Tensorflow does not support circular padding")
        
        result = tf.nn.conv1d(input_tensor, weight, stride=stride, padding=padding_mode, dilation_rate=dilation)

        if bias:
            result = tf.nn.bias_add(result, bias_val)
        
        result = tf.nn.relu(result)
        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=0)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros',
        "weight": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "bias_val": np.array([0.1, 0.2], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()