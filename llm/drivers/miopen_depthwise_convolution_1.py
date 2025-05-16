import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1,))
    padding = input_dict.get("padding", (0,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.nn.functional.conv2d(
        input_tensor,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

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
        stride = input_dict.get("stride", (1,))
        padding = input_dict.get("padding", (0,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        strides = [1, stride[0], stride[1], 1]
        dilations = [1, dilation[0], dilation[1], 1]
        padding_val = 'VALID'
        if padding != (0,0):
            padding_val = 'SAME'
        
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        weight = tf.expand_dims(weight, axis=0)

        result = tf.nn.depthwise_conv2d(
            input_tensor,
            weight,
            strides=strides,
            padding=padding_val,
            dilations=dilations
        )

        if bias is not None:
            result = tf.nn.bias_add(result, bias)
        
        result = tf.squeeze(result, axis=0)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 32, 32).astype(np.float32),
        "weight": np.random.rand(3, 1, 3, 3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 3,
    }
    torch_result = torch_version({"input": input_data["input"][None, ...], "weight": input_data["weight"], "bias": input_data["bias"], "stride": input_data["stride"], "padding": input_data["padding"], "dilation": input_data["dilation"], "groups": input_data["groups"]})
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 16, 16).astype(np.float32),
        "weight": np.random.rand(1, 1, 3, 3).astype(np.float32),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
    }

    torch_result = torch_version({"input": input_data["input"][None, ...], "weight": input_data["weight"], "stride": input_data["stride"], "padding": input_data["padding"], "dilation": input_data["dilation"], "groups": input_data["groups"]})
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")