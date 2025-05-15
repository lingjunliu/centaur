import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig, default_qat_qconfig

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
    qconfig = default_qat_qconfig

    if not cpu:
        input_tensor = input_tensor.cuda()

    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0)
    
    conv_relu = nniqat.ConvReLU1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig)
    conv_relu.weight = nn.Parameter(torch.tensor(input_dict["weight"]))
    if bias:
        conv_relu.bias = nn.Parameter(torch.tensor(input_dict["bias_val"]))

    if not cpu:
        conv_relu = conv_relu.cuda()
        conv_relu.weight = nn.Parameter(conv_relu.weight.cuda())
        if bias:
            conv_relu.bias = nn.Parameter(conv_relu.bias.cuda())
    
    result = conv_relu(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

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
        padding_mode = input_dict.get("padding_mode", 'VALID')

        if padding > 0:
           input_tensor = tf.pad(input_tensor, [[0, 0], [padding, padding]], "CONSTANT")
           padding_mode = 'VALID'
        
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)

        weight = tf.transpose(weight, perm=[2, 0, 1])
        weight = tf.expand_dims(weight, axis=0)

        if len(input_tensor.shape) == 1:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        elif len(input_tensor.shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        if groups > 1:
            input_splits = tf.split(input_tensor, num_or_size_splits=groups, axis=2)
            weight_splits = tf.split(weight, num_or_size_splits=groups, axis=3)
            outputs = []
            for i in range(groups):
                conv = tf.nn.conv1d(input_splits[i], weight_splits[i], stride=stride, padding=padding_mode, data_format='NWC')
                outputs.append(conv)
            conv_result = tf.concat(outputs, axis=2)
        else:
            conv_result = tf.nn.conv1d(input_tensor, weight, stride=stride, padding=padding_mode, data_format='NWC')
                
        if bias:
            bias_val = tf.constant(input_dict["bias_val"], dtype=tf.float32)
            conv_result = tf.nn.bias_add(conv_result, bias_val, data_format='NWC')

        if dilation > 1:
            expanded_result = []
            for i in range(conv_result.shape[0]):
                row = []
                for j in range(conv_result.shape[1]):
                    row.append(conv_result[i][j].numpy())
                    for k in range(dilation-1):
                        row.append(0.0)
                expanded_result.append(row)
            
            expanded_result = tf.constant(expanded_result)
            conv_result = tf.expand_dims(expanded_result, axis=0)

        relu_result = tf.nn.relu(conv_result)
        result = relu_result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_length = 10
    in_channels = 3
    out_channels = 5
    kernel_size = 3

    input_data = {
        "input": np.random.rand(1, input_length, in_channels).astype(np.float32),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros',
        "weight": np.random.rand(out_channels, kernel_size, in_channels).astype(np.float32),
        "bias_val": np.random.rand(out_channels).astype(np.float32)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()