import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.intrinsic.quantized as nniq

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    weight = torch.tensor(input_dict["weight"], dtype=torch.float32)
    bias = torch.tensor(input_dict["bias"], dtype=torch.float32) if "bias" in input_dict else None

    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    conv_relu_3d = nniq.ConvReLU3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

    with torch.no_grad():
        conv_relu_3d.weight.data.copy_(weight)
        if conv_relu_3d.bias is not None and bias is not None:
            conv_relu_3d.bias.data.copy_(bias)

    result = conv_relu_3d(input_tensor)

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
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None

        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride, stride)
        
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        if isinstance(padding, int):
            padding = (padding, padding, padding)

        if padding_mode == 'zeros':
          if padding != (0, 0, 0):
            input_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]], [0, 0]] )
          padding_tf = 'VALID'
        else:
          raise ValueError(f"Padding mode {padding_mode} not supported in tensorflow. Only zero padding supported")

        # Transpose weight for TF conv3d
        weight_transposed = tf.transpose(weight, perm=[2, 3, 4, 1, 0])  # Corrected for 3D convolution

        result = tf.nn.conv3d(input_tensor, weight_transposed, strides=[1, stride[0], stride[1], stride[2], 1], padding=padding_tf)

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = tf.nn.relu(result)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": (3, 3, 3),
        "padding": (1, 1, 1),
        "stride": (2, 2, 2),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()