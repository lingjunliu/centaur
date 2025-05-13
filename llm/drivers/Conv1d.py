import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.quantized import Conv1d

    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()

    if bias:
        weights = torch.tensor(input_dict["weights"])
        bias_tensor = torch.tensor(input_dict["bias_tensor"])
    else:
        weights = torch.tensor(input_dict["weights"])
        bias_tensor = None
    
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])

    if not cpu:
        weights = weights.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()
        if bias_tensor is not None:
            bias_tensor = bias_tensor.cuda()

    result = Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    
    with torch.no_grad():
        result.weight.data.copy_(weights)
        if bias_tensor is not None:
            result.bias.data.copy_(bias_tensor)
        result.scale = scale
        result.zero_point = zero_point

    result.eval()

    with torch.no_grad():
        result = result(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        weights = tf.constant(input_dict["weights"], dtype=tf.float32)
        if bias:
            bias_tensor = tf.constant(input_dict["bias_tensor"], dtype=tf.float32)
        else:
            bias_tensor = None

        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict["zero_point"], dtype=tf.float32)

        input_tensor_reshaped = tf.reshape(input_tensor, [1, input_tensor.shape[0], 1])

        weights_reshaped = tf.reshape(weights, [kernel_size, 1, out_channels])

        if padding > 0:
             input_tensor_reshaped = tf.pad(input_tensor_reshaped, [[0, 0], [padding, padding], [0, 0]])
             
        result = tf.nn.conv1d(input_tensor_reshaped, filters=weights_reshaped, stride=stride, padding="VALID", data_format="NWC", dilations=dilation)

        if bias_tensor is not None:
           result = tf.add(result, bias_tensor)

        result = tf.reshape(result, [result.shape[1],out_channels])
        result = result.numpy()

        result = scale.numpy() * (result - zero_point.numpy())

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "weights": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "bias_tensor": np.array([0.5], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32),
        "zero_point": np.array([0], dtype=np.float32),
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()