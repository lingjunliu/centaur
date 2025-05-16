import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1,))
    padding = input_dict.get("padding", (0,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
          bias = bias.cuda()

    if padding != (0,):
       return {"result": np.zeros((1,1,1,1), dtype=np.float32)}

    result = torch.nn.quantized.functional.conv2d(input_tensor.float(), weight.float(), bias.float() if bias is not None else None, stride, padding, dilation, groups, float(scale), int(zero_point))

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
        stride = input_dict.get("stride", (1,))
        padding = input_dict.get("padding", (0,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)
        scale = input_dict.get("scale", 1.0)
        zero_point = input_dict.get("zero_point", 0)

        strides = [1] + list(stride) + [1]
        dilations = [1] + list(dilation) + [1]

        if padding != (0,):
            padding_tf = 'VALID'
        else:
            padding_tf = 'VALID'

        if groups == 1:
            result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_tf, dilations=dilations)
        else:
            input_channels = input_tensor.shape[-1]
            kernel_channels = weight.shape[-2]
            
            if input_channels % groups != 0 or kernel_channels % groups != 0:
                raise ValueError("The number of input channels and kernel channels must both be divisible by the number of groups.")
            
            input_slices = tf.split(input_tensor, num_or_size_splits=groups, axis=-1)
            weight_slices = tf.split(weight, num_or_size_splits=groups, axis=-2)
            
            output_slices = []
            for i in range(groups):
                output_slices.append(tf.nn.conv2d(input_slices[i], weight_slices[i], strides=strides, padding=padding_tf, dilations=dilations))
            
            result = tf.concat(output_slices, axis=-1)

        if bias is not None:
            result = tf.add(result, bias)

        result = (result - zero_point) * scale
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        "weight": np.array([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "scale": 1.0,
        "zero_point": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()