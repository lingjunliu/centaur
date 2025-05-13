import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

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

    result = torch.nn.functional.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

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

        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(padding, int):
            padding = (padding,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2

        strides_tf = [1, stride[0], stride[1], 1]
        padding_tf = "VALID" if padding == (0, 0) else "SAME"

        result = tf.nn.depthwise_conv2d(
            input_tensor,
            weight,
            strides=strides_tf,
            padding=padding_tf,
        )

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 5, 5, 1).astype(np.float32),
        "weight": np.random.rand(3, 3, 1, 1).astype(np.float32),
        "bias": np.random.rand(1).astype(np.float32),
        "stride": (1,1),
        "padding": (0,0),
        "dilation": (1,1),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()