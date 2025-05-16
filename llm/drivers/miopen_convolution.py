import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = None if "bias" not in input_dict else torch.tensor(input_dict["bias"])
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
        bias,
        stride,
        padding,
        dilation,
        groups
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
        bias = None if "bias" not in input_dict else tf.constant(input_dict["bias"])
        stride = input_dict.get("stride", (1,))
        padding = input_dict.get("padding", (0,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)

        stride = [1] + list(stride) + [1]
        dilation = list(dilation)

        if isinstance(padding, str):
            padding_tf = padding.upper()
        else:
            padding_tf = "VALID"
            if padding != (0,):
                p = padding[0] if len(padding) == 1 else (padding[0], padding[0])
                input_tensor = tf.pad(input_tensor, [[0, 0], [p[0], p[1]], [p[0], p[1]], [0, 0]])
                padding_tf = "VALID"

        input_channels = int(input_tensor.shape[-1])
        kernel_input_channels = int(weight.shape[1])
        
        if groups > 1:
            if input_channels % groups != 0:
                raise ValueError("The number of input channels must be evenly divisible by the number of groups.")
            if weight.shape[1] != input_tensor.shape[1]:
                raise ValueError("The number of kernel input channels should be equal to the number of input channels.")

            input_slices = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
            weight_slices = tf.split(weight, num_or_size_splits=groups, axis=0)

            output_slices = []
            for i in range(groups):
                output_slices.append(tf.nn.conv2d(input_slices[i], weight_slices[i], strides=stride, padding=padding_tf, dilations=dilation))
            result = tf.concat(output_slices, axis=3)
        else:
            result = tf.nn.conv2d(input_tensor, weight, strides=stride, padding=padding_tf, dilations=dilation)
            
        if bias is not None:
            result = tf.nn.bias_add(result, bias)

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
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()