import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    gamma = torch.tensor(input_dict["gamma"])
    beta = torch.tensor(input_dict["beta"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    qconfig = input_dict.get("qconfig", QConfig(activation=torch.nn.Identity, weight=torch.nn.Identity))

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        gamma = gamma.cuda()
        beta = beta.cuda()

    conv_bn = nniqat.ConvBn3d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"],
                               stride=stride, padding=padding, dilation=dilation, groups=groups,
                               bias=bias is not None, padding_mode=padding_mode, qconfig=qconfig)

    conv_bn.weight.data = weight
    if bias is not None:
      conv_bn.bias.data = bias

    conv_bn.bn.running_mean.data = running_mean
    conv_bn.bn.running_var.data = running_var
    conv_bn.bn.weight.data = gamma
    conv_bn.bn.bias.data = beta
    conv_bn.bn.eps = eps
    conv_bn.bn.momentum = momentum

    result = conv_bn(input_tensor)

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
        gamma = tf.constant(input_dict["gamma"])
        beta = tf.constant(input_dict["beta"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports padding_mode='zeros'")

        strides = [1, stride, stride, stride, 1] if isinstance(stride, int) else [1, stride[0], stride[1], stride[2], 1]
        dilations = [1, dilation, dilation, dilation, 1] if isinstance(dilation, int) else [1, dilation[0], dilation[1], dilation[2], 1]

        padding_val = 'VALID' if padding == 0 else 'SAME' if padding != 0 else None

        if isinstance(padding, int):
            padding_tf = padding
        else:
            padding_tf = 'VALID' if padding == (0, 0, 0) else 'SAME'
            if padding != (0,0,0):
                raise ValueError("Tensorflow only supports padding 0 or SAME")

        weight_shape = weight.shape.as_list()
        kernel_size = input_dict["kernel_size"]

        if weight_shape[0] != kernel_size:
            weight = tf.transpose(weight, perm=[2,3,4,1,0])


        conv = tf.nn.conv3d(input_tensor, weight, strides=strides, padding=padding_val, dilations=dilations, data_format='NCDHW')

        if bias is not None:
            conv = tf.nn.bias_add(conv, value=bias, data_format='NCDHW')
        
        mean, variance = tf.nn.moments(conv, axes=[2,3,4], keepdims=True)
        normalized_input = (conv - mean) / tf.sqrt(variance + eps)
        bn = gamma * normalized_input + beta

        result = bn.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 3,
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "running_mean": np.random.rand(5).astype(np.float32),
        "running_var": np.random.rand(5).astype(np.float32),
        "gamma": np.random.rand(5).astype(np.float32),
        "beta": np.random.rand(5).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()