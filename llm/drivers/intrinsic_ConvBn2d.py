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
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic as nni

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])

    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    batch_norm_eps = input_dict.get("batch_norm_eps", 1e-05)
    batch_norm_momentum = input_dict.get("batch_norm_momentum", 0.1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv_bn = nni.ConvBn2d(
        input_dict["in_channels"],
        input_dict["out_channels"],
        input_dict["kernel_size"],
        eps=batch_norm_eps,
        momentum=batch_norm_momentum
    )

    conv_bn.conv.stride = (stride, stride)
    conv_bn.conv.padding = (padding, padding)
    conv_bn.conv.dilation = (dilation, dilation)
    conv_bn.conv.groups = groups
    conv_bn.conv.bias = True
    conv_bn.weight.data = weight
    conv_bn.bias.data = bias
    conv_bn.running_mean.data = running_mean
    conv_bn.running_var.data = running_var

    result = conv_bn(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

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
        batch_norm_eps = input_dict.get("batch_norm_eps", 1e-05)
        batch_norm_momentum = input_dict.get("batch_norm_momentum", 0.1)

        kernel_size = input_dict["kernel_size"]
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]

        weight = tf.transpose(tf.reshape(weight, [out_channels, in_channels // groups, *kernel_size]), perm=[2, 3, 1, 0])
        
        if isinstance(padding, int):
            padding_tf = 'VALID' if padding == 0 else 'SAME'
        else:
            padding_tf = 'VALID'

        conv = tf.nn.conv2d(input_tensor, weight, strides=[1, stride, stride, 1], padding=padding_tf, dilations=[1, dilation, dilation, 1])
        conv = tf.nn.bias_add(conv, bias)

        scale = tf.math.rsqrt(running_var + batch_norm_eps)
        gamma = tf.ones([out_channels], dtype=tf.float32)
        beta = tf.zeros([out_channels], dtype=tf.float32)
        
        mean = running_mean
        variance = running_var

        bn = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, batch_norm_eps)
        result = bn

        result = result.numpy()
    
    return {"result": result}

def main():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "running_mean": np.random.rand(16).astype(np.float32),
        "running_var": np.random.rand(16).astype(np.float32),
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()