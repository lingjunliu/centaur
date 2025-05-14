import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0]))) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0])), dtype=tf.float32) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        if isinstance(stride, int):
            strides = [1, stride, stride, 1]
        else:
            strides = [1, stride[0], stride[1], 1]

        if isinstance(dilation, int):
            dilations = [1, dilation, dilation, 1]
        else:
            dilations = [1, dilation[0], dilation[1], 1]

        if padding == 0:
            padding_mode = 'VALID'
        elif isinstance(padding, int):
            padding_mode = 'SAME'
        else:
            padding_mode = 'VALID'

        input_shape = input_tensor.shape.as_list()
        weight_shape = weight.shape.as_list()

        if groups > 1:
            if input_shape[1] % groups != 0:
                raise ValueError("Number of groups must divide the number of input channels.")
            channels = input_shape[1] // groups
            weight = tf.reshape(weight, [weight_shape[0], weight_shape[1], channels, groups, weight_shape[3]])
            weight = tf.transpose(weight, [3, 0, 1, 2, 4])
            weight = tf.reshape(weight, [weight_shape[0] * groups, weight_shape[1], channels, weight_shape[3]])

        result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_mode, dilations=dilations, data_format='NCHW')

        if bias is not None:
            result = tf.nn.bias_add(tf.transpose(result, perm=[0, 2, 3, 1]), bias, data_format='NHWC')
            result = tf.transpose(result, perm=[0, 3, 1, 2])

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(8, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(8).astype(np.float32),
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