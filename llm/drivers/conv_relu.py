import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.intrinsic.quantized.modules.conv_relu import ConvReLU2d

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    conv_relu = ConvReLU2d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"], stride, padding, dilation, groups, bias is not None, padding_mode)
    with torch.no_grad():
        conv_relu.weight[:] = weight
        if bias is not None:
            conv_relu.bias[:] = bias
    
    input_tensor = input_tensor.float()
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
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'VALID' if input_dict.get("padding", 0) == 0 else 'SAME')

        input_shape = input_dict["input"].shape
        weight_shape = input_dict["weight"].shape
        
        input_tensor = tf.reshape(input_tensor, (input_shape[0], input_shape[2], input_shape[3], input_shape[1]))

        weight = tf.reshape(weight, (weight_shape[2], weight_shape[3], weight_shape[1], weight_shape[0]))

        if padding_mode == 'SAME':
            result_conv = tf.nn.conv2d(
                input_tensor, weight, strides=[1, stride, stride, 1], padding=padding_mode, dilations=[1, dilation, dilation, 1]
            )
        else:
            if isinstance(padding, int):
                pad_total = padding * 2
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
                input_padded = tf.pad(input_tensor, paddings, "CONSTANT")
            else:
                paddings = [[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]]
                input_padded = tf.pad(input_tensor, paddings, "CONSTANT")
            result_conv = tf.nn.conv2d(
                input_padded, weight, strides=[1, stride, stride, 1], padding='VALID', dilations=[1, dilation, dilation, 1]
            )

        if bias is not None:
            result_conv = tf.nn.bias_add(result_conv, bias)
        result = tf.nn.relu(result_conv)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 5,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()