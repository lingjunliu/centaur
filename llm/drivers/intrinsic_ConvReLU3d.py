import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = None
    if "bias" in input_dict:
        bias = torch.tensor(input_dict["bias"])
    
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
          bias = bias.cuda()

    conv3d = nn.Conv3d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"],
                       stride=stride, padding=padding, dilation=dilation, groups=groups, bias=(bias is not None), padding_mode=padding_mode)
    conv3d.weight.data = weight
    if bias is not None:
      conv3d.bias.data = bias
    
    relu = nn.ReLU()
    
    result = conv3d(input_tensor)
    result = relu(result)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

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
        bias = None
        if "bias" in input_dict:
            bias = tf.constant(input_dict["bias"])

        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        
        padding_mode = 'VALID'
        
        if input_dict.get("padding_mode", 'zeros') == 'zeros' and padding != 0:
          pad_size = padding
          input_tensor = tf.pad(input_tensor, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT')
          padding_mode = 'VALID'
        elif padding != 0:
            padding_mode = 'SAME'
            
        strides = [1, stride, stride, stride, 1]
        dilations = [1, dilation, dilation, dilation, 1]

        input_shape = input_dict["input"].shape
        kernel_shape = input_dict["kernel_size"]
        
        data_format = 'NDHWC'
        
        if len(input_shape) == 5 and input_shape[-1] == input_dict["in_channels"]:
            pass
        else:
            data_format = 'NCDHW'
            input_tensor = tf.transpose(input_tensor, perm=[0, 4, 1, 2, 3])
            weight = tf.transpose(weight, perm=[4, 3, 0, 1, 2])

        result = tf.nn.conv3d(input_tensor, weight, strides=strides, padding=padding_mode, dilations=dilations, data_format=data_format)
        
        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format=data_format)
        
        result = tf.nn.relu(result)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 10, 10, 10, 3).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": (3, 3, 3),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 5, 5, 5, 3).astype(np.float32),
        "weight": np.random.rand(4, 3, 2, 2, 2).astype(np.float32),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 1,
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