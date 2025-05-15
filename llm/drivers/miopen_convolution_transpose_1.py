import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
    
    result = torch.nn.functional.conv_transpose2d(
        input_tensor,
        weight,
        bias=None, #torch api does not accept bias
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation
    )
    
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
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", False)

        input_shape = input_tensor.shape.as_list()
        weight_shape = weight.shape.as_list()

        batch_size = input_shape[0]
        in_channels = input_shape[1]
        in_height = input_shape[2]
        in_width = input_shape[3]

        out_channels = weight_shape[0]
        
        if in_channels % groups != 0:
            raise ValueError('in_channels ({}) must be divisible by groups ({})'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels ({}) must be divisible by groups ({})'.format(out_channels, groups))
        
        filter_height = kernel_size[0]
        filter_width = kernel_size[1]
        
        if groups > 1:
            raise ValueError("Groups > 1 are not supported in tensorflow for conv_transpose")
        
        if dilation != 1:
            raise ValueError("Dilation != 1 are not supported in tensorflow for conv_transpose")

        if isinstance(stride, int):
            stride_x = stride_y = stride
        else:
            stride_x, stride_y = stride

        if isinstance(padding, int):
            padding_x = padding_y = padding
        else:
            padding_x, padding_y = padding

        if isinstance(output_padding, int):
            output_padding_x = output_padding_y = output_padding
        else:
            output_padding_x, output_padding_y = output_padding
        
        out_height = (in_height - 1) * stride_y - 2 * padding_y + filter_height + output_padding_y
        out_width = (in_width - 1) * stride_x - 2 * padding_x + filter_width + output_padding_x

        output_shape = [batch_size, out_channels, out_height, out_width]
        strides = [1, stride_y, stride_x, 1]
        
        if padding_x == 0 and padding_y == 0:
            padding_arg = 'VALID'
        else:
            padding_arg = 'SAME'

        weight_reordered = tf.transpose(weight, perm=[2, 3, 1, 0])

        result = tf.nn.conv2d_transpose(
            input_tensor,
            weight_reordered,
            output_shape=output_shape,
            strides=strides,
            padding=padding_arg
        )
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 4, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 4, 3, 3).astype(np.float32),
        "kernel_size": (3, 3),
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "dilation": 1,
        "groups": 1,
        "bias": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()