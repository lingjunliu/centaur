import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.qat import ConvBn1d
    from torch.quantization import QConfig, default_qconfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])

    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    qconfig = input_dict.get("qconfig")
    if qconfig is None:
        qconfig = default_qconfig
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv_bn = ConvBn1d(input_tensor.shape[1], weight.shape[0], weight.shape[1], eps=eps, momentum=momentum, padding=padding, stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode, qconfig=qconfig)

    conv_bn.weight = torch.nn.Parameter(weight)
    conv_bn.bias = torch.nn.Parameter(bias)
    conv_bn.running_mean = running_mean
    conv_bn.running_var = running_var
    conv_bn.eval()
    with torch.no_grad():
        result = conv_bn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

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
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding != 0:
            paddings = [[0, 0], [0, 0], [padding, padding]]
            input_tensor = tf.pad(input_tensor, paddings, mode='CONSTANT')

        input_tensor = tf.expand_dims(input_tensor, axis=1)

        weight = tf.transpose(weight, perm=[2, 1, 0])
        weight = tf.expand_dims(weight, axis=0)
        strides = [1, 1, stride, 1]
        dilations = [1, 1, dilation, 1]
        conv_out = tf.nn.conv2d(input_tensor, weight, strides=strides, padding='VALID', data_format='NHWC', dilations=dilations)

        conv_out = tf.squeeze(conv_out, axis=1)
        
        mean = running_mean
        variance = running_var

        inv = tf.math.rsqrt(variance + eps)
        normalized = (conv_out - mean) * inv
        
        gamma = tf.ones_like(bias)
        beta = bias

        output = gamma * normalized + beta

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    import torch
    from torch.quantization import QConfig, default_qconfig

    input_channels = 3
    output_channels = 5
    kernel_size = 2
    seq_len = 10
    batch_size = 2

    input_data = {
        "input": np.random.rand(batch_size, input_channels, seq_len).astype(np.float32),
        "weight": np.random.rand(output_channels, input_channels, kernel_size).astype(np.float32),
        "bias": np.random.rand(output_channels).astype(np.float32),
        "running_mean": np.random.rand(output_channels).astype(np.float32),
        "running_var": np.random.rand(output_channels).astype(np.float32),
        "qconfig": default_qconfig
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()