import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    dilation = input_dict.get("dilation", 1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        
    result = torch.nn.functional.conv_transpose3d(
        input_tensor,
        weight,
        bias=None,
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
        groups = input_dict.get("groups", 1)
        dilation = input_dict.get("dilation", 1)
        
        input_shape = input_dict["input"].shape
        batch_size = input_shape[0]
        in_channels = input_shape[1]
        in_depth = input_shape[2]
        in_height = input_shape[3]
        in_width = input_shape[4]
        
        weight_shape = input_dict["weight"].shape
        out_channels = weight_shape[0]
        
        if isinstance(stride, int):
            stride_depth = stride_height = stride_width = stride
        else:
            stride_depth, stride_height, stride_width = stride
        
        if isinstance(padding, int):
            padding_depth = padding_height = padding_width = padding
        else:
            padding_depth, padding_height, padding_width = padding
        
        if isinstance(output_padding, int):
            output_padding_depth = output_padding_height = output_padding_width = output_padding
        else:
            output_padding_depth, output_padding_height, output_padding_width = output_padding
        
        # Calculate output shape
        out_depth = (in_depth - 1) * stride_depth - 2 * padding_depth + dilation * (kernel_size[0] - 1) + output_padding_depth + 1
        out_height = (in_height - 1) * stride_height - 2 * padding_height + dilation * (kernel_size[1] - 1) + output_padding_height + 1
        out_width = (in_width - 1) * stride_width - 2 * padding_width + dilation * (kernel_size[2] - 1) + output_padding_width + 1

        output_shape = (batch_size, out_channels, out_depth, out_height, out_width)

        # Reshape input and filters for tf.nn.conv3d
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])  # NCDHW -> NDHWC
        weight = tf.transpose(weight, perm=[2, 3, 4, 1, 0])  # D,H,W,in_channels,out_channels -> DHWinCout

        # Perform transposed convolution
        result = tf.nn.conv3d_transpose(
            input=input_tensor,
            filters=weight,
            output_shape=tf.constant(output_shape, dtype=tf.int32),
            strides=(1, stride_depth, stride_height, stride_width, 1),
            padding="VALID",
            data_format="NDHWC"
        )
        
        result = tf.transpose(result, perm=[0, 4, 1, 2, 3]) # NDHWC -> NCDHW
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    in_channels = 3
    out_channels = 2
    kernel_size = (2, 2, 2)

    input_data = {
        "input": np.random.rand(1, in_channels, 4, 4, 4).astype(np.float32),
        "weight": np.random.rand(out_channels, in_channels, *kernel_size).astype(np.float32),
        "kernel_size": kernel_size,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "output_padding": (0, 0, 0),
        "groups": 1,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()