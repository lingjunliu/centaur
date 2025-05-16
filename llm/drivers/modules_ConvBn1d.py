import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.intrinsic.qat.modules import ConvBn1d
    from torch.quantization import QConfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    gamma = torch.tensor(input_dict["gamma"])
    beta = torch.tensor(input_dict["beta"])

    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        gamma = gamma.cuda()
        beta = beta.cuda()

    qconfig = QConfig(activation=torch.quantization.default_observer, weight=torch.quantization.default_observer)

    module = ConvBn1d(
        in_channels=input_tensor.shape[1],
        out_channels=weight.shape[0],
        kernel_size=weight.shape[2],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
        padding_mode=padding_mode,
        qconfig=qconfig
    )

    module.weight = torch.nn.Parameter(weight)
    module.bias = torch.nn.Parameter(bias)
    module.bn.running_mean = running_mean
    module.bn.running_var = running_var
    module.bn.weight = torch.nn.Parameter(gamma)
    module.bn.bias = torch.nn.Parameter(beta)
    module.bn.eps = eps
    module.bn.momentum = momentum

    result = module(input_tensor)

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
        gamma = tf.constant(input_dict["gamma"])
        beta = tf.constant(input_dict["beta"])

        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow only supports padding_mode='zeros'")

        if groups != 1:
            raise ValueError("TensorFlow Conv1D only supports groups=1")

        # TensorFlow Conv1D expects input with shape [batch, width, in_channels]
        # and weight with shape [width, in_channels, out_channels]
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1])
        weight = tf.transpose(weight, perm=[2, 1, 0])

        result = tf.nn.conv1d(
            input_tensor,
            filters=weight,
            stride=stride,
            padding='VALID',
            dilation_rate=dilation
        )

        result = tf.nn.bias_add(result, bias)

        mean = running_mean
        variance = running_var
        
        result = tf.nn.batch_normalization(
            result,
            mean=mean,
            variance=variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=eps
        )

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01
    input_channels = 3
    out_channels = 5
    kernel_size = 3
    batch_size = 2
    seq_len = 10

    input_data = {
        "input": np.random.rand(batch_size, input_channels, seq_len).astype(np.float32),
        "weight": np.random.rand(out_channels, input_channels, kernel_size).astype(np.float32),
        "bias": np.random.rand(out_channels).astype(np.float32),
        "running_mean": np.random.rand(out_channels).astype(np.float32),
        "running_var": np.random.rand(out_channels).astype(np.float32),
        "gamma": np.random.rand(out_channels).astype(np.float32),
        "beta": np.random.rand(out_channels).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "eps": 1e-05,
        "momentum": 0.1,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()