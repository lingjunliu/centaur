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
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig, default_qat_qconfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    qconfig = default_qat_qconfig
    conv_bn_relu = nniqat.ConvBnReLU3d(
        input_tensor.shape[1],
        weight.shape[0],
        weight.shape[2:5],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        eps=eps,
        momentum=momentum,
        bias=(bias is not None),
        qconfig=qconfig
    )
    
    conv_bn_relu.weight.data = weight
    if bias is not None:
        conv_bn_relu.bias.data = bias

    conv_bn_relu.eval()

    result = conv_bn_relu(input_tensor)

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
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)

        if isinstance(stride, int):
          stride = [stride, stride, stride]
        if isinstance(dilation, int):
          dilation = [dilation, dilation, dilation]
        if isinstance(padding, int):
          padding = [[padding, padding], [padding, padding], [padding, padding]]
        elif isinstance(padding, tuple):
          padding = [[padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]]]

        if padding_mode == 'zeros':
            padded_input = tf.pad(input_tensor, [[0,0], [0,0], [padding[0][0], padding[0][1]], [padding[1][0], padding[1][1]], [padding[2][0], padding[2][1]]])
        else:
            raise ValueError("padding_mode != 'zeros' is not supported yet")

        input_shape = padded_input.shape
        kernel_shape = weight.shape

        if input_shape[1] != kernel_shape[1] * groups:
            raise ValueError("Input channels must be divisible by the number of groups")

        if groups == 1:
            conv = tf.nn.conv3d(padded_input, weight, strides=[1, 1, stride[0], stride[1], stride[2]], padding='VALID', data_format='NCDHW')
        else:
            raise ValueError("groups != 1 is not supported yet")

        if bias is not None:
            conv = tf.nn.bias_add(conv, bias, data_format='NCDHW')
        
        axes = [0]
        reduction_indices = [2,3,4]

        mean, variance = tf.nn.moments(conv, axes=reduction_indices, keepdims=True)
        bn = tf.nn.batch_normalization(conv, mean, variance, offset=None, scale=None, variance_epsilon=eps)

        relu = tf.nn.relu(bn)

        result = relu.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 5, 5, 5).astype(np.float32),
        "weight": np.random.rand(6, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "padding_mode": 'zeros',
        "eps": 1e-05,
        "momentum": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()