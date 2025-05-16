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
    num_features = input_dict["num_features"]
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    qconfig = input_dict.get("qconfig", None)
    freeze_bn = input_dict.get("freeze_bn", False)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv_bn_relu = nniqat.ConvBnReLU2d(
        input_dict["in_channels"],
        input_dict["out_channels"],
        input_dict["kernel_size"],
        stride=input_dict.get("stride", 1),
        padding=input_dict.get("padding", 0),
        dilation=input_dict.get("dilation", 1),
        groups=input_dict.get("groups", 1),
        bias=True,
        padding_mode=input_dict.get("padding_mode", 'zeros'),
        qconfig=input_dict["qconfig"] if "qconfig" in input_dict else None
    )
    conv_bn_relu.weight = nn.Parameter(weight)
    conv_bn_relu.bias = nn.Parameter(bias)
    conv_bn_relu.bn.running_mean = running_mean
    conv_bn_relu.bn.running_var = running_var
    conv_bn_relu.bn.eps = eps
    conv_bn_relu.bn.momentum = momentum
    conv_bn_relu.bn.weight = nn.Parameter(torch.ones_like(running_var))
    conv_bn_relu.bn.bias = nn.Parameter(torch.zeros_like(running_var))

    conv_bn_relu.eval()
    with torch.no_grad():
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
        input_tensor = tf.constant(input_dict["input"])
        num_features = input_dict["num_features"]
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        qconfig = input_dict.get("qconfig", None)
        freeze_bn = input_dict.get("freeze_bn", False)
        inplace = input_dict.get("inplace", False)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        
        if padding_mode != 'zeros':
            raise ValueError("Tensorflow does not support padding_mode other than zeros.")

        channels = input_dict["input"].shape[3]
        kernel_size = input_dict["kernel_size"]
        
        if isinstance(kernel_size, int):
            kernel_size_eff = (kernel_size, kernel_size)
        else:
            kernel_size_eff = kernel_size
        
        if isinstance(stride, int):
            stride_eff = (stride, stride)
        else:
            stride_eff = stride
            
        if isinstance(dilation, int):
            dilation_eff = (dilation, dilation)
        else:
            dilation_eff = dilation
        
        if padding == 0:
            padding_eff = 'VALID'
        else:
            padding_eff = 'SAME'

        gamma = tf.ones_like(running_var)
        beta = tf.zeros_like(running_var)

        # Reshape for broadcasting
        running_mean = tf.reshape(running_mean, [1, 1, 1, num_features])
        running_var = tf.reshape(running_var, [1, 1, 1, num_features])
        gamma = tf.reshape(gamma, [1, 1, 1, num_features])
        beta = tf.reshape(beta, [1, 1, 1, num_features])
        
        bn_output = tf.nn.batch_normalization(
            input_tensor,
            mean=running_mean,
            variance=running_var,
            offset=beta,
            scale=gamma,
            variance_epsilon=eps
        )
        
        weight = tf.transpose(weight, perm=[2, 3, 0, 1])

        result = tf.nn.relu(tf.nn.conv2d(bn_output, weight, strides=[1, stride_eff[0], stride_eff[1], 1], padding=padding_eff) + bias)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "padding_mode": 'zeros',
        "num_features": 8,
        "weight": np.random.rand(8, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(8).astype(np.float32),
        "running_mean": np.random.rand(8).astype(np.float32),
        "running_var": np.random.rand(8).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "qconfig": torch.ao.quantization.default_qat_qconfig
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()