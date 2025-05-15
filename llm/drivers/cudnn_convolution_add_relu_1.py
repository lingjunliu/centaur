import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    padding = input_dict.get("padding", (0,))
    stride = input_dict.get("stride", (1,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.nn.functional.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)
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
        bias = tf.constant(input_dict["bias"])
        padding = input_dict.get("padding", (0,))
        stride = input_dict.get("stride", (1,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)

        if len(padding) == 1:
            padding = 'VALID' if padding[0] == 0 else 'SAME'
        elif len(padding) == 2:
            if padding[0] == 0 and padding[1] == 0:
                padding = 'VALID'
            else:
                padding = 'SAME'
        else:
            padding = 'VALID'
            
        input_shape = input_tensor.shape
        weight_shape = weight.shape

        in_channels = input_shape[-1]
        out_channels = weight_shape[0]

        if in_channels % groups != 0:
            raise ValueError("Input channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("Output channels must be divisible by groups")
        
        result = tf.nn.conv2d(input_tensor, weight, strides=[1, stride[0], stride[1], 1], padding=padding, dilations=[1, dilation[0], dilation[1], 1], data_format='NHWC')
        result = tf.nn.bias_add(result, bias, data_format='NHWC')
        result = tf.nn.relu(result)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    in_channels = 3
    out_channels = 4

    input_data = {
        "input": np.random.rand(1, in_channels, 5, 5).astype(np.float32),
        "weight": np.random.rand(out_channels, in_channels, 3, 3).astype(np.float32),
        "bias": np.random.rand(out_channels).astype(np.float32),
        "padding": (1, 1),
        "stride": (1, 1),
        "dilation": (1, 1),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    
    tf_input = np.transpose(input_data["input"], (0, 2, 3, 1))
    tf_weight = np.transpose(input_data["weight"], (2, 3, in_channels, out_channels))

    tf_result = tensorflow_version({
        "input": tf_input,
        "weight": tf_weight,
        "bias": input_data["bias"],
        "padding": input_data["padding"],
        "stride": input_data["stride"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"]
    })
    
    torch_result_transposed = np.transpose(torch_result["result"], (0, 2, 3, 1))

    assert np.allclose(torch_result_transposed, tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()