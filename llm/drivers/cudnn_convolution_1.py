import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = torch.tensor(input_dict["weight"])
    bias_tensor = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    padding = input_dict.get("padding", (0,))
    stride = input_dict.get("stride", (1,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)
    benchmark = input_dict.get("benchmark", False)
    deterministic = input_dict.get("deterministic", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        if bias_tensor is not None:
            bias_tensor = bias_tensor.cuda()

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    
    result = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)

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
        weight_tensor = tf.constant(input_dict["weight"])
        bias_tensor = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        padding = input_dict.get("padding", (0,0))
        stride = input_dict.get("stride", (1,1))
        dilation = input_dict.get("dilation", (1,1))
        groups = input_dict.get("groups", 1)

        if len(padding) == 1:
            padding = [padding[0], padding[0]]
        if len(stride) == 1:
            stride = [stride[0], stride[0]]
        if len(dilation) == 1:
            dilation = [dilation[0], dilation[0]]

        padding_tf = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]

        input_tensor = tf.pad(input_tensor, padding_tf, "CONSTANT")

        stride_tf = [1, stride[0], stride[1], 1]
        dilation_tf = [1, dilation[0], dilation[1], 1]

        input_shape = input_tensor.shape
        weight_shape = weight_tensor.shape
        
        if groups > 1:
            input_channels = int(input_shape[-1])
            kernel_channels = int(weight_shape[1])

            if input_channels % groups != 0 or weight_shape[1] * groups != input_channels:
                raise ValueError("Number of input or output channels not divisible by groups.")

            def convolution_group(i):
                input_group = input_tensor[:, :, :, i * input_channels // groups:(i + 1) * input_channels // groups]
                kernel_group = weight_tensor[:, :, :, i * kernel_channels // groups:(i + 1) * kernel_channels // groups]
                kernel_group = tf.transpose(kernel_group, perm=[2, 3, 1, 0])
                return tf.nn.conv2d(input_group, kernel_group, strides=stride_tf, padding='VALID', dilations=dilation_tf)

            convolutions = [convolution_group(i) for i in range(groups)]
            result = tf.concat(convolutions, axis=-1)

        else:
            if input_shape[-1] % weight_shape[1] != 0:
                # Pad the input channels to be divisible by the weight channels
                padding_channels = weight_shape[1] - (input_shape[-1] % weight_shape[1])
                padding_before = 0
                padding_after = padding_channels
                padding_tensor = [[0, 0], [0, 0], [0, 0], [padding_before, padding_after]]
                input_tensor = tf.pad(input_tensor, padding_tensor, "CONSTANT")
                input_shape = input_tensor.shape
           
            weight_tensor = tf.transpose(weight_tensor, perm=[2, 3, 1, 0])
            result = tf.nn.conv2d(input_tensor, weight_tensor, strides=stride_tf, padding='VALID', dilations=dilation_tf)

        if bias_tensor is not None:
            result = tf.nn.bias_add(result, bias_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "padding": (2,2),
        "stride": (1,1),
        "dilation": (1,1),
        "groups": 1,
        "benchmark": False,
        "deterministic": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()