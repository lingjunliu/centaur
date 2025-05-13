import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.qat.modules import ConvBn1d
    from torch.quantization import QConfig
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    weight = torch.tensor(input_dict["weight"], dtype=torch.float32)
    bias = torch.tensor(input_dict["bias"], dtype=torch.float32)
    running_mean = torch.tensor(input_dict["running_mean"], dtype=torch.float32)
    running_var = torch.tensor(input_dict["running_var"], dtype=torch.float32)
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    qconfig = input_dict.get("qconfig", QConfig(activation=torch.quantization.default_observer, weight=torch.quantization.default_observer))

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv_bn = ConvBn1d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"],
                       stride=stride, padding=padding, dilation=dilation, groups=groups,
                       padding_mode=padding_mode, qconfig=qconfig)
    
    conv_bn.weight.data = weight.unsqueeze(0)
    if conv_bn.bias is not None:
        conv_bn.bias.data = bias
    
    conv_bn.bn.running_mean.data = running_mean
    conv_bn.bn.running_var.data = running_var
    conv_bn.bn.eps = eps
    conv_bn.bn.momentum = momentum

    result = conv_bn(input_tensor.unsqueeze(0).unsqueeze(0))

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze(0).squeeze(0).detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=2)
        weight = tf.expand_dims(weight, axis=0)
        weight = tf.expand_dims(weight, axis=2)

        if padding_mode == 'zeros':
            padding_tf = 'VALID' if padding == 0 else 'SAME'
        else:
            raise ValueError("TensorFlow does not support padding modes other than 'zeros'")
        
        result = tf.nn.conv1d(input_tensor, weight, stride=stride, padding=padding_tf, data_format='NWC', dilations=dilation)

        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format='NWC')
        
        scale = tf.math.rsqrt(running_var + eps)
        
        mean = running_mean
        variance = running_var

        result = (result - mean) * scale
        
        result = result * weight + bias
        
        result = result.numpy()

    return {"result": result.squeeze(0).squeeze(1)}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "weight": np.array([0.5], dtype=np.float32),
        "bias": np.array([0.1], dtype=np.float32),
        "running_mean": np.array([0.2], dtype=np.float32),
        "running_var": np.array([0.3], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "padding_mode": 'zeros',
        "eps": 1e-05,
        "momentum": 0.1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()