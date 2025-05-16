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
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.quantized as nniq

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1, 1, 1))
    padding = input_dict.get("padding", (0, 0, 0))
    dilation = input_dict.get("dilation", (1, 1, 1))
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    conv_relu = nniq.ConvReLU3d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"], 
                                stride=stride, padding=padding, dilation=dilation, groups=groups, 
                                padding_mode=padding_mode, bias=(bias is not None))
    
    with torch.no_grad():
        conv_relu.weight.data = weight
        if bias is not None:
            conv_relu.bias.data = bias

    result = conv_relu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None
        stride = input_dict.get("stride", (1, 1, 1))
        padding = input_dict.get("padding", (0, 0, 0))
        dilation = input_dict.get("dilation", (1, 1, 1))
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'VALID')

        if isinstance(stride, int):
            stride = [1, stride, stride, stride, 1]
        else:
            stride = [1, stride[0], stride[1], stride[2], 1]

        if padding_mode == 'zeros':
            if isinstance(padding, int):
                padding = 'VALID' if padding == 0 else 'SAME'
            else:
                if padding == (0, 0, 0):
                    padding = 'VALID'
                else:
                    padding = 'SAME'
        else:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")

        if groups == 1:
            conv = tf.nn.conv3d(
                input=tf.expand_dims(input_tensor, axis=0),
                filters=weight,
                strides=stride,
                padding=padding,
                data_format="NCDHW",
                dilations=dilation
            )
        else:
            raise NotImplementedError("Groups > 1 is not supported for tf.nn.conv3d")

        if bias is not None:
            conv = tf.nn.bias_add(conv, bias, data_format="NCDHW")

        relu = tf.nn.relu(conv)

        result = relu.numpy()[0]

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 16, 16, 16).astype(np.float32),
        "weight": np.random.rand(8, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(8).astype(np.float32),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "groups": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()