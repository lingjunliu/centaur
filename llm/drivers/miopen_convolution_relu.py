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
        
        if isinstance(stride, int):
            stride = [stride] * 2
        if isinstance(dilation, int):
            dilation = [dilation] * 2

        strides = [1] + list(stride) + [1]
        dilations = [1] + list(dilation) + [1]

        padding_mode = 'VALID'
        if padding != (0,):
            padding_mode = 'SAME'

        result = tf.nn.conv2d(
            input=input_tensor,
            filters=weight,
            strides=strides,
            padding=padding_mode,
            dilations=dilations,
        )

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

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
        "padding": (2, 2),
        "dilation": (1, 1),
        "groups": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version({"input": input_data["input"].transpose(0, 2, 3, 1),
                                 "weight": input_data["weight"].transpose(2, 3, 1, 0),
                                 "bias": input_data["bias"],
                                 "stride": input_data["stride"],
                                 "padding": input_data["padding"],
                                 "dilation": input_data["dilation"],
                                 "groups": input_data["groups"]})

    assert np.allclose(torch_result["result"], tf_result["result"].transpose(0,3,1,2), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()