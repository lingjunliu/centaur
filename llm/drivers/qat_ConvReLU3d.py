import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.intrinsic.qat import ConvReLU3d
    from torch.ao.quantization import QConfig, default_qat_qconfig

    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()

    qconfig = default_qat_qconfig
    conv_relu = ConvReLU3d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=groups,
                                bias=bias, padding_mode=padding_mode, qconfig=qconfig)

    weight = torch.tensor(input_dict["weight"])
    if not cpu:
        weight = weight.cuda()
    conv_relu.weight = torch.nn.Parameter(weight)
    if bias:
        bias_val = torch.tensor(input_dict["bias_val"])
        if not cpu:
            bias_val = bias_val.cuda()
        conv_relu.bias = torch.nn.Parameter(bias_val)

    if not cpu:
        conv_relu = conv_relu.cuda()

    input_tensor = input_tensor.permute(0, 4, 1, 2, 3)

    result = conv_relu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        
        if isinstance(kernel_size, int):
          kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        else:
          kernel_size_tuple = kernel_size
        
        if isinstance(stride, int):
          stride_tuple = (stride, stride, stride)
        else:
          stride_tuple = stride

        if isinstance(dilation, int):
            dilation_tuple = (dilation, dilation, dilation)
        else:
            dilation_tuple = dilation
        
        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'
        
        strides_tf = [1, stride_tuple[0], stride_tuple[1], stride_tuple[2], 1]
        dilations_tf = [1, dilation_tuple[0], dilation_tuple[1], dilation_tuple[2], 1]

        weight = tf.transpose(weight, perm=[2, 3, 4, 1, 0])

        output = tf.nn.conv3d(input_tensor, weight, strides=strides_tf, padding=padding_tf, dilations=dilations_tf)
        
        if bias:
            bias_val = tf.constant(input_dict["bias_val"], dtype=tf.float32)
            output = tf.nn.bias_add(output, bias_val)
            
        output = tf.nn.relu(output)
        
        output = tf.transpose(output, perm=[0, 4, 1, 2, 3])
        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_shape = (1, 10, 10, 10, 3)
    in_channels = 3
    out_channels = 5
    kernel_size = 3

    input_data = {
        "input": np.random.rand(*input_shape).astype(np.float32),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "weight": np.random.rand(kernel_size, kernel_size, kernel_size, in_channels, out_channels).astype(np.float32),
        "bias_val": np.random.rand(out_channels).astype(np.float32),
    }

    input_data["weight"] = np.transpose(input_data["weight"], (4, 3, 0, 1, 2))

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()