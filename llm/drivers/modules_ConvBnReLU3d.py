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
    from torch.quantization import QConfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])

    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    freeze_bn = input_dict.get("freeze_bn", False)
    if input_dict.get("qconfig") is None:
        qconfig = QConfig(activation=torch.quantization.default_observer, weight=torch.quantization.default_observer)
    else:
        qconfig = input_dict.get("qconfig")

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv_bn_relu = nniqat.ConvBnReLU3d(
        input_tensor.shape[1],
        weight.shape[0],
        weight.shape[2],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        eps=eps,
        momentum=momentum,
        freeze_bn=freeze_bn,
        qconfig=qconfig
    )

    conv_bn_relu.weight = nn.Parameter(weight)
    conv_bn_relu.bias = nn.Parameter(bias)
    conv_bn_relu.running_mean = running_mean
    conv_bn_relu.running_var = running_var

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
        bias = tf.constant(input_dict["bias"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])

        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        freeze_bn = input_dict.get("freeze_bn", False)
        qconfig = input_dict.get("qconfig", None)

        if padding_mode == 'zeros':
            padding_mode_tf = 'VALID'
            if padding > 0:
                input_tensor_padded = tf.pad(input_tensor, [[0, 0], [0, 0], [padding, padding], [padding, padding], [padding, padding]])
                input_tensor = input_tensor_padded
        else:
            raise ValueError("Padding mode {} not supported in TensorFlow".format(padding_mode))

        strides = [1, stride, stride, stride, 1]
        dilations = [1, dilation, dilation, dilation, 1]

        # TF Conv3D expects the channel dimension to be last
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])
        weight = tf.transpose(weight, perm=[2, 3, 4, 1, 0])
            
        conv = tf.nn.conv3d(input_tensor, weight, strides=strides, padding=padding_mode_tf, dilations=dilations)

        conv = tf.nn.bias_add(conv, bias, data_format='NDHWC')

        # Batch norm needs the channel dimension to be last too.
        mean, variance = tf.nn.moments(conv, axes=[0,1,2,3], keepdims=True)

        if freeze_bn:
             bn = tf.nn.batch_normalization(conv, running_mean, running_var, None, None, eps)
        else:
            bn = tf.nn.batch_normalization(conv, mean, variance, None, None, eps)

        relu = tf.nn.relu(bn)

        # Transpose back to NCDHW
        result = tf.transpose(relu, perm=[0, 4, 1, 2, 3]).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 5, 5, 5).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "running_mean": np.random.rand(5).astype(np.float32),
        "running_var": np.random.rand(5).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "padding_mode": 'zeros',
        "eps": 1e-05,
        "momentum": 0.1,
        "freeze_bn": False,
        "qconfig": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()