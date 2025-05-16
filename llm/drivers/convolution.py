import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
        bias = torch.tensor(bias)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)
    if isinstance(dilation, int):
        dilation = (dilation,)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.convolution(input_tensor, weight, bias, stride, padding, dilation, False, (0,), groups)

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
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        input_shape = input_tensor.shape
        weight_shape = weight.shape
        input_tensor = tf.reshape(input_tensor, (1, input_shape[1], input_shape[2], input_shape[3]))
        weight = tf.reshape(weight, (weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[0]))

        if isinstance(stride, int):
            strides = [1, stride, stride, 1]
        else:
            strides = [1, stride[0], stride[1], 1]

        if isinstance(dilation, int):
            dilations = [1, dilation, dilation, 1]
        else:
            dilations = [1, dilation[0], dilation[1], 1]

        if isinstance(padding, int):
            padding_type = "VALID" if padding == 0 else "SAME"
        else:
            if padding[0] == 0 and padding[1] == 0:
                padding_type = "VALID"
            else:
                padding_type = "SAME"
        
        result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_type, dilations=dilations)

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = tf.reshape(result, result.shape[1:])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]]]], dtype=np.float32),
        "weight": np.array([[[[[0.1, 0.2]]]]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]]]], dtype=np.float32),
        "weight": np.array([[[[[0.1, 0.2]]]]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]]]], dtype=np.float32),
        "weight": np.array([[[[[0.1, 0.2]]]]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()