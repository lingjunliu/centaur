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

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])

    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    freeze_bn = input_dict.get("freeze_bn", False)
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
    
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
    conv.weight.data = weight
    conv.bias.data = bias
    bn = nn.BatchNorm1d(out_channels, eps=eps, momentum=momentum)
    bn.running_mean.data = running_mean
    bn.running_var.data = running_var
    bn.weight.data = torch.ones_like(running_mean)
    bn.bias.data = torch.zeros_like(running_mean)

    if qconfig is not None:
        conv_bn_relu = nniqat.ConvBnReLU1d(conv, bn, nn.ReLU())
        conv_bn_relu.freeze_bn = freeze_bn
        conv_bn_relu.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    else:
        conv_bn_relu = nniqat.ConvBnReLU1d(conv, bn, nn.ReLU())
        conv_bn_relu.freeze_bn = freeze_bn
        conv_bn_relu.qconfig = None

    result = conv_bn_relu(input_tensor.unsqueeze(0))

    if not cpu:
        result = result.cpu()
    
    return {"result": result.squeeze(0).numpy()}

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
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)

        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        freeze_bn = input_dict.get("freeze_bn", False)
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=2)

        weight = tf.expand_dims(weight, axis=1)
        weight = tf.expand_dims(weight, axis=0)
        
        conv = tf.nn.conv1d(input_tensor, weight, stride=stride, padding='VALID', data_format='NWC', dilations=dilation)
        conv = tf.nn.bias_add(conv, bias, data_format='NWC')

        mean = running_mean
        variance = running_var
        gamma = tf.ones_like(running_mean, dtype=tf.float32)
        beta = tf.zeros_like(running_mean, dtype=tf.float32)

        conv = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, eps)
        conv = tf.nn.relu(conv)

        result = conv.numpy().flatten()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "weight": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "bias": np.array([0.1], dtype=np.float32),
        "running_mean": np.array([0.2], dtype=np.float32),
        "running_var": np.array([0.3], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "eps": 1e-05,
        "momentum": 0.1,
        "freeze_bn": False,
        "qconfig": "qconfig"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()