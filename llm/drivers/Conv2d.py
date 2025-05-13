import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.quantized as nnq

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    qconv = nnq.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    with torch.no_grad():
        qconv.weight[:] = weight
        if bias is not None:
            qconv.bias[:] = bias

    result = qconv(input_tensor)

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
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        
        strides = [1, stride, stride, 1]
        dilations = [1, dilation, dilation, 1]
        padding_method = 'VALID'
        if padding > 0:
            padding_method = 'SAME'

        if groups == 1:
            weight = tf.transpose(weight, perm=[2, 3, 1, 0])
            result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_method, dilations=dilations)
        else:
            input_channels = input_dict["in_channels"]
            output_channels = input_dict["out_channels"]
            kernel_size = input_dict["kernel_size"]

            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            weight_shape = (kernel_size[0], kernel_size[1], input_channels // groups, output_channels)
            weight = tf.reshape(weight, weight_shape)
            weight = tf.transpose(weight, perm=[2, 3, 0, 1])

            splits = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
            weight_splits = tf.split(weight, num_or_size_splits=groups, axis=3)

            convs = []
            for i in range(groups):
                conv = tf.nn.conv2d(splits[i], weight_splits[i], strides=strides, padding=padding_method, dilations=dilations)
                convs.append(conv)
            result = tf.concat(convs, axis=3)

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "in_channels": 3,
        "out_channels": 2,
        "kernel_size": 3,
        "input": np.random.rand(1, 3, 28, 28).astype(np.float32),
        "weight": np.random.rand(2, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(2).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)

    input_data["input"] = np.transpose(input_data["input"], (0, 2, 3, 1))

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()