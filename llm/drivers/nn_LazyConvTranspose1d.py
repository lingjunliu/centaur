import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    dilation = input_dict.get("dilation", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    layer = torch.nn.LazyConvTranspose1d(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)

    if not cpu:
        layer = layer.cuda()

    result = layer(input_tensor)
    
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
        input_tensor = tf.constant(input_dict["input"])
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        dilation = input_dict.get("dilation", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        in_channels = input_tensor.shape[1]
        
        if in_channels % groups != 0:
            groups = 1

        kernel_shape = (kernel_size, in_channels // groups, out_channels)

        fan_in = np.prod(kernel_shape[:-1])
        limit = np.sqrt(6 / fan_in)
        kernel_val = np.random.uniform(-limit, limit, size=kernel_shape).astype(np.float32)
        kernel = tf.Variable(kernel_val)

        if bias:
            bias_val = np.random.uniform(-limit, limit, size=(out_channels,)).astype(np.float32)
            bias_var = tf.Variable(bias_val)
        else:
            bias_var = None

        input_tensor_expanded = tf.expand_dims(input_tensor, axis=0)
        input_tensor_expanded = tf.expand_dims(input_tensor_expanded, axis=-1)
        
        kernel_expanded = tf.transpose(kernel, perm=[0, 2, 1])
        kernel_expanded = tf.expand_dims(kernel_expanded, axis=-1)
        
        output_shape_val = (1, (input_tensor.shape[-1] - 1) * stride + kernel_size - 2 * padding + output_padding, out_channels, 1)
        output_shape = tf.constant(output_shape_val, dtype=tf.int32)
        
        input_filter_depth = input_tensor_expanded.shape[3]
        kernel_filter_depth = kernel_expanded.shape[2]
        if input_filter_depth != kernel_filter_depth:
            raise ValueError(f"Input filter depth ({input_filter_depth}) must match kernel filter depth ({kernel_filter_depth})")
        
        result = tf.nn.conv2d_transpose(
            input_tensor_expanded,
            kernel_expanded,
            output_shape=output_shape,
            strides=[1, stride, 1, 1],
            padding="VALID"
        )
        
        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=-1)

        if bias_var is not None:
            result = tf.nn.bias_add(result, bias_var, data_format="NWC")
            result = tf.transpose(result, perm=[1, 0])

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 5).astype(np.float32),
        "out_channels": 3,
        "kernel_size": 3,
    }
    
    input_data["groups"] = 1

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()