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
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    relu = input_dict.get("relu", True)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    qconfig = input_dict.get("qconfig", default_qat_qconfig)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv_bn_relu = nniqat.ConvBnReLU2d(
        in_channels=input_tensor.shape[1],
        out_channels=weight.shape[0],
        kernel_size=weight.shape[2:4],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        qconfig=qconfig,
        bias = True if bias is not None else False
    )
    conv_bn_relu.weight = nn.Parameter(weight)
    if bias is not None:
        conv_bn_relu.bias = nn.Parameter(bias)
    conv_bn_relu.bn.running_mean = running_mean
    conv_bn_relu.bn.running_var = running_var
    conv_bn_relu.bn.eps = eps
    conv_bn_relu.bn.momentum = momentum

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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        relu = input_dict.get("relu", True)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports zero padding")
        
        strides = [1, stride, stride, 1]
        dilations = [1, dilation, dilation, 1]
        
        if isinstance(padding, int):
            padding_tf = 'VALID' if padding == 0 else 'SAME'
        else:
            padding_tf = 'VALID' if padding[0] == 0 else 'SAME'
        
        # Transpose weight to [height, width, in_channels, out_channels]
        weight_transposed = tf.transpose(weight, perm=[2, 3, 1, 0])

        # Group the conv2d
        if groups == 1:
          conv = tf.nn.conv2d(input_tensor, weight_transposed, strides=strides, padding=padding_tf, dilations=dilations)
        else:
            # Split the input and weights into groups
            input_splits = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
            weight_splits = tf.split(weight_transposed, num_or_size_splits=groups, axis=2)

            # Perform grouped convolution
            conv_list = []
            for i in range(groups):
                conv_i = tf.nn.conv2d(input_splits[i], weight_splits[i], strides=strides, padding=padding_tf, dilations=dilations)
                conv_list.append(conv_i)

            # Concatenate the results
            conv = tf.concat(conv_list, axis=3)

        if bias is not None:
            conv = tf.nn.bias_add(conv, bias)
        
        scale = tf.math.rsqrt(running_var + eps)
        
        # Reshape gamma and beta for broadcasting
        gamma = tf.reshape(scale, [1, 1, 1, -1]) # Reshape it to NHWC format

        beta = tf.reshape(bias if bias is not None else tf.zeros([weight.shape[0]], dtype=weight.dtype), [1, 1, 1, -1]) # Use conv bias as beta if it exist otherwise create beta
        
        bn = tf.nn.batch_normalization(conv, running_mean, running_var, beta, tf.ones_like(gamma), eps)

        if relu:
            bn = tf.nn.relu(bn)

        result = bn.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(8, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(8).astype(np.float32),
        "running_mean": np.random.rand(8).astype(np.float32),
        "running_var": np.random.rand(8).astype(np.float32),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()