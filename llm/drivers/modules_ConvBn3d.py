import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.qat.modules import ConvBn3d
    from torch.quantization import QConfig
    from torch import nn

    input_tensor = torch.tensor(input_dict["input"])
    bn_weight = torch.tensor(input_dict["bn_weight"])
    bn_bias = torch.tensor(input_dict["bn_bias"])
    bn_running_mean = torch.tensor(input_dict["bn_running_mean"])
    bn_running_var = torch.tensor(input_dict["bn_running_var"])

    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    qconfig = QConfig(activation=nn.Identity, weight=nn.Identity)

    if not cpu:
        input_tensor = input_tensor.cuda()
        bn_weight = bn_weight.cuda()
        bn_bias = bn_bias.cuda()
        bn_running_mean = bn_running_mean.cuda()
        bn_running_var = bn_running_var.cuda()

    conv_bn = ConvBn3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, qconfig=qconfig)

    conv_bn.bn.weight = torch.nn.Parameter(bn_weight)
    conv_bn.bn.bias = torch.nn.Parameter(bn_bias)
    conv_bn.bn.running_mean = bn_running_mean
    conv_bn.bn.running_var = bn_running_var

    result = conv_bn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        bn_weight = tf.constant(input_dict["bn_weight"])
        bn_bias = tf.constant(input_dict["bn_bias"])
        bn_running_mean = tf.constant(input_dict["bn_running_mean"])
        bn_running_var = tf.constant(input_dict["bn_running_var"])

        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])

        if padding_mode == 'zeros':
            if padding != (0, 0, 0):
                input_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]], [0, 0]])

            W = tf.random.normal(shape=(kernel_size[0], kernel_size[1], kernel_size[2], in_channels, out_channels))

            result_conv = tf.nn.conv3d(input_tensor, W, strides=[1, stride[0], stride[1], stride[2], 1], padding='VALID')

            if not bias:
                result_conv = result_conv
            else:
                b = tf.zeros(shape=(out_channels,))
                result_conv = tf.nn.bias_add(result_conv, b)


            scale = bn_weight / tf.sqrt(bn_running_var + 1e-5)
            result = scale * (result_conv - bn_running_mean) + bn_bias
            result = tf.transpose(result, perm=[0, 4, 1, 2, 3])


        else:
            raise ValueError("Padding mode not supported")

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "bn_weight": np.random.rand(3).astype(np.float32),
        "bn_bias": np.random.rand(3).astype(np.float32),
        "bn_running_mean": np.random.rand(3).astype(np.float32),
        "bn_running_var": np.random.rand(3).astype(np.float32),
        "in_channels": 3,
        "out_channels": 3,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros'
    }


    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()