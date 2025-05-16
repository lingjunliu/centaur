import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.qat.modules as intrinsic_qat_modules
    from torch.quantization import QConfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None

    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    batchnorm_module = torch.nn.BatchNorm3d(input_dict["weight"].shape[0])
    batchnorm_module.weight = torch.nn.Parameter(torch.tensor(input_dict["bn_weight"]))
    batchnorm_module.bias = torch.nn.Parameter(torch.tensor(input_dict["bn_bias"]))
    batchnorm_module.running_mean = torch.tensor(input_dict["bn_running_mean"])
    batchnorm_module.running_var = torch.tensor(input_dict["bn_running_var"])
    batchnorm_module.eps = input_dict.get("bn_eps", 1e-05)
    batchnorm_module.momentum = input_dict.get("bn_momentum", 0.1)
    batchnorm_module.eval()

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
        batchnorm_module.cuda()

    qconfig = QConfig(activation=torch.quantization.default_observer, weight=torch.quantization.default_observer)
    conv_bn_relu = intrinsic_qat_modules.ConvBnReLU3d(input_dict["input"].shape[1], input_dict["weight"].shape[0], kernel_size=(input_dict["weight"].shape[2], input_dict["weight"].shape[3], input_dict["weight"].shape[4]), qconfig=qconfig)
    conv_bn_relu.weight = torch.nn.Parameter(weight)
    if bias is not None:
        conv_bn_relu.bias = torch.nn.Parameter(bias)
    conv_bn_relu.bn = batchnorm_module
    conv_bn_relu.stride = stride if isinstance(stride, int) else tuple(stride)
    conv_bn_relu.padding = padding if isinstance(padding, int) else tuple(padding)
    conv_bn_relu.dilation = dilation if isinstance(dilation, int) else tuple(dilation)
    conv_bn_relu.groups = groups
    conv_bn_relu.padding_mode = padding_mode

    with torch.no_grad():
        result = conv_bn_relu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None

        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        bn_weight = tf.constant(input_dict["bn_weight"])
        bn_bias = tf.constant(input_dict["bn_bias"])
        bn_running_mean = tf.constant(input_dict["bn_running_mean"])
        bn_running_var = tf.constant(input_dict["bn_running_var"])
        bn_eps = input_dict.get("bn_eps", 1e-05)

        if padding_mode == 'zeros':
            padding_tf = 'VALID' if padding == 0 else 'SAME'
        else:
            raise ValueError(f"Unsupported padding mode: {padding_mode}")

        input_shape = input_dict["input"].shape
        weight_shape = input_dict["weight"].shape

        if padding == 0:
            output_shape = (input_shape[2] - weight_shape[2] + 1,
                              input_shape[3] - weight_shape[3] + 1,
                              input_shape[4] - weight_shape[4] + 1)
        else:
            output_shape = (input_shape[2], input_shape[3], input_shape[4])

        # Reshape for TensorFlow's expected input format
        input_tensor = tf.reshape(input_tensor, (1, input_shape[2], input_shape[3], input_shape[4], input_shape[1]))
        weight = tf.reshape(weight, (weight_shape[2], weight_shape[3], weight_shape[4], weight_shape[1], weight_shape[0]))

        if groups == 1:
            conv = tf.nn.conv3d(input_tensor, weight, strides=[1, stride, stride, stride, 1] if isinstance(stride, int) else [1, stride[0], stride[1], stride[2], 1], padding=padding_tf, dilations=[1, dilation, dilation, dilation, 1] if isinstance(dilation, int) else [1, dilation[0], dilation[1], dilation[2], 1])
        else:
            raise ValueError("Groups > 1 not supported in TensorFlow.")

        # Reshape bias if it exists
        if bias is not None:
            bias = tf.reshape(bias, (weight_shape[0],))
            conv = tf.nn.bias_add(conv, bias)

        # Reshape gamma and beta for batch normalization
        gamma = tf.reshape(bn_weight, [weight_shape[0]])
        beta = tf.reshape(bn_bias, [weight_shape[0]])

        # Calculate batch normalization
        bn = tf.nn.batch_normalization(
            conv,
            bn_running_mean,
            bn_running_var,
            beta,
            gamma,
            bn_eps,
        )

        # Apply ReLU
        result = tf.nn.relu(bn)

        # Reshape back to match PyTorch output
        result = tf.reshape(result, (1, weight_shape[0], output_shape[0], output_shape[1], output_shape[2])).numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "bn_weight": np.random.rand(5).astype(np.float32),
        "bn_bias": np.random.rand(5).astype(np.float32),
        "bn_running_mean": np.random.rand(5).astype(np.float32),
        "bn_running_var": np.random.rand(5).astype(np.float32),
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "bn_eps": 1e-05,
        "bn_momentum": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()