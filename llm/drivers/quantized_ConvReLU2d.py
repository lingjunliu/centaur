import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    conv_relu = torch.nn.intrinsic.quantized.ConvReLU2d(input_tensor.shape[1], weight.shape[0], kernel_size=weight.shape[2], stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)
    
    with torch.no_grad():
        conv_relu.weight.data.copy_(weight)
        conv_relu.bias.data.copy_(bias)

    result = conv_relu(input_tensor)
    
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
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if isinstance(stride, int):
            strides = [1, stride, stride, 1]
        else:
            strides = [1, stride[0], stride[1], 1]
        
        if isinstance(dilation, int):
            rates = [1, dilation, dilation, 1]
        else:
            rates = [1, dilation[0], dilation[1], 1]

        if isinstance(padding, int):
            padding_val = 'VALID' if padding == 0 else 'SAME'
        else:
            padding_val = 'VALID' if padding == (0,0) else 'SAME'

        if groups == 1:
            conv = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_val, dilations=rates)
        else:
            input_splits = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
            weight_splits = tf.split(weight, num_or_size_splits=groups, axis=3)
            
            conv_splits = []
            for i in range(groups):
                conv_splits.append(tf.nn.conv2d(input_splits[i], weight_splits[i], strides=strides, padding=padding_val, dilations=rates))
            
            conv = tf.concat(conv_splits, axis=3)
        
        result = tf.nn.relu(tf.nn.bias_add(conv, bias))
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(6, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "padding_mode": 'zeros'
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()