import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict.get("in_channels", 0)
    out_channels = input_dict.get("out_channels", 0)
    kernel_size = input_dict.get("kernel_size", 1)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    lazy_conv3d = torch.nn.LazyConv3d(
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    if not cpu:
        lazy_conv3d = lazy_conv3d.cuda()
    
    result = lazy_conv3d(input_tensor)
    
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
        in_channels = input_dict.get("in_channels", 0)
        out_channels = input_dict.get("out_channels", 0)
        kernel_size = input_dict.get("kernel_size", 1)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (1, stride, stride, stride, 1)
        else:
            stride = (1, stride[0], stride[1], stride[2], 1)
        
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        
        if isinstance(dilation, int):
            dilation = (1, dilation, dilation, dilation, 1)
        else:
            dilation = (1, dilation[0], dilation[1], dilation[2], 1)

        
        input_shape = input_tensor.shape
        if len(input_shape) == 5:
            batch_size, channels, depth, height, width = input_shape
        elif len(input_shape) == 4:
            batch_size, depth, height, width = input_shape
            channels = 1
            input_tensor = tf.expand_dims(input_tensor, axis=1)
        elif len(input_shape) == 3:
            batch_size, height, width = input_shape
            depth = 1
            channels = 1
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        else:
            batch_size = 1
            channels = 1
            depth = 1
            height = 1
            width = 1
            input_tensor = tf.reshape(input_tensor, (batch_size, channels, depth, height, width))

        
        kernel_shape = (kernel_size[0], kernel_size[1], kernel_size[2], input_tensor.shape[1] // groups, out_channels)
        
        kernel = tf.random.normal(kernel_shape, dtype=tf.float32)
        
        if bias:
          bias_shape = (out_channels,)
          bias_tf = tf.random.normal(bias_shape, dtype=tf.float32)
        else:
          bias_tf = None

        
        
        result = tf.nn.conv3d(
            input=input_tensor,
            filters=kernel,
            strides=stride,
            padding="VALID",
            data_format="NCDHW",
            dilations=dilation
        )
        
        if bias_tf is not None:
          result = tf.nn.bias_add(result, bias_tf, data_format="NCDHW")
        
        pad_list = [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]]]
        result = tf.pad(result, pad_list)
        

        result = result.numpy()
        
        
    
    return {"result": result}

def main():
    A_TOL = 0.1
    input_data = {
        "input": np.random.rand(2, 3, 16, 16, 16).astype(np.float32),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()