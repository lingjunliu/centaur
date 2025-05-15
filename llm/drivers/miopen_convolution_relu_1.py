import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1,))
    padding = input_dict.get("padding", (0,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)
    relu = input_dict.get("relu", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = F.conv2d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    if relu:
        result = torch.relu(result)

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
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        stride = input_dict.get("stride", (1,))
        padding = input_dict.get("padding", (0,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)
        relu = input_dict.get("relu", False)

        if len(input_tensor.shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        if len(weight.shape) == 3:
            weight = tf.expand_dims(weight, axis=0)

        if isinstance(stride, int):
            stride = [stride]
        if isinstance(dilation, int):
            dilation = [dilation]
            
        if len(stride) == 1:
            stride = [1, stride[0], stride[0], 1]
        elif len(stride) == 2:
            stride = [1, stride[0], stride[1], 1]
        
        if len(dilation) == 1:
             dilation = [1, dilation[0], dilation[0], 1]
        elif len(dilation) == 2:
             dilation = [1, dilation[0], dilation[1], 1]
            
        if padding == (0,):
             padding = "VALID"
        else:
             padding = "VALID"

        input_shape = input_tensor.shape
        weight_shape = weight.shape

        in_channels = int(input_shape[-1])
        out_channels = int(weight_shape[0])

        
        if groups > 1:
            in_channels_per_group = in_channels // groups
            weight = tf.reshape(weight, (weight_shape[0], in_channels_per_group, weight_shape[2], weight_shape[3]))
            weight = tf.transpose(weight, perm=[2, 3, 1, 0])
            result = tf.nn.depthwise_conv2d(input_tensor, weight, strides=stride, padding=padding, dilations=dilation)
        else:
            weight = tf.transpose(weight, perm=[2, 3, int(weight_shape[1]), int(weight_shape[0])])
            result = tf.nn.conv2d(input_tensor, weight, strides=stride, padding=padding, dilations=dilation)

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        if relu:
            result = tf.nn.relu(result)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(6, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "relu": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()