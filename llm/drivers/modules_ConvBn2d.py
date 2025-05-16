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
    from torch.quantization import QConfig, default_qconfig

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

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    qconfig = default_qconfig
    module = nniqat.ConvBn2d(input_dict['in_channels'], input_dict['out_channels'], input_dict['kernel_size'], 
                               stride=stride, padding=padding, dilation=dilation, groups=groups, 
                               bias=True, eps=eps, momentum=momentum, qconfig=qconfig)
    module.weight = nn.Parameter(weight)
    module.bias = nn.Parameter(bias)
    module.running_mean = running_mean
    module.running_var = running_var
    module.eval()

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

        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        freeze_bn = input_dict.get("freeze_bn", False)
        
        if isinstance(stride, int):
            stride = [1, stride, stride, 1]
        else:
            stride = [1, stride[0], stride[1], 1]

        if isinstance(padding, int):
            padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
        else:
            padding = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
        
        
        input_channels = input_dict["in_channels"]
        
        if input_dict["groups"] > 1:
            weight = tf.transpose(weight, perm=[2, 3, 1, 0])
        else:
            weight = tf.transpose(weight, perm=[3, 2, 1, 0])

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])
        input_tensor = tf.pad(input_tensor, padding, "CONSTANT")
        conv = tf.nn.conv2d(input_tensor, weight, strides=stride, padding='VALID', data_format="NHWC")
        
        bn_output = tf.nn.batch_normalization(
            conv,
            mean=running_mean,
            variance=running_var,
            offset=bias,
            scale=tf.ones_like(running_var),
            variance_epsilon=eps
        )

        result = bn_output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(6, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "running_mean": np.random.rand(6).astype(np.float32),
        "running_var": np.random.rand(6).astype(np.float32),
        "in_channels": 3,
        "out_channels": 6,
        "kernel_size": 5,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "eps": 1e-05,
        "momentum": 0.1,
        "freeze_bn": False
    }
    
    input_data['input'] = np.random.rand(1, input_data["in_channels"], 32, 32).astype(np.float32)
    input_data['weight'] = np.random.rand(input_data["out_channels"], input_data["in_channels"] // input_data["groups"], input_data["kernel_size"], input_data["kernel_size"]).astype(np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()