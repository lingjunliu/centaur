import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1,))
    padding = input_dict.get("padding", (0,))
    output_padding = input_dict.get("output_padding", (0,))
    groups = input_dict.get("groups", 1)
    dilation = input_dict.get("dilation", (1,))
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    input_tensor = input_tensor.permute(0, 3, 1, 2)
    if bias is not None:
        result = torch.nn.functional.conv_transpose2d(input_tensor, weight, bias, stride, padding, output_padding, groups, dilation)
    else:
        result = torch.nn.functional.conv_transpose2d(input_tensor, weight, None, stride, padding, output_padding, groups, dilation)
    result = result.permute(0, 2, 3, 1)

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
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        stride = input_dict.get("stride", (1,))
        padding = input_dict.get("padding", (0,))
        output_padding = input_dict.get("output_padding", (0,))
        groups = input_dict.get("groups", 1)
        dilation = input_dict.get("dilation", (1,))

        input_shape = input_tensor.shape
        weight_shape = weight.shape

        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        if len(stride) == 1:
            stride = (stride[0], stride[0])
        if len(padding) == 1:
            padding = (padding[0], padding[0])
        if len(output_padding) == 1:
            output_padding = (output_padding[0], output_padding[0])
        if len(dilation) == 1:
            dilation = (dilation[0], dilation[0])
        
        strides = [1, stride[0], stride[1], 1]
        dilations = [1, dilation[0], dilation[1], 1]

        if padding[0] == 0 and padding[1] == 0:
            padding_mode = 'VALID'
        else:
            padding_mode = 'SAME'
            
        # Calculate output shape correctly
        output_height = (input_shape[1] - 1) * strides[1] + weight_shape[0] - 2 * padding[0] + output_padding[0]
        output_width = (input_shape[2] - 1) * strides[2] + weight_shape[1] - 2 * padding[1] + output_padding[1]
        output_shape = (input_shape[0], output_height, output_width, weight_shape[2])
        output_shape = tf.TensorShape(output_shape)

        result = tf.nn.conv2d_transpose(input_tensor, weight, output_shape, strides=strides, padding=padding_mode, data_format='NHWC', dilations = dilations)
        
        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format='NHWC')
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 1e-3

    input_channels = 3
    output_channels = 3
    kernel_size = 3
    batch_size = 1
    input_height = 5
    input_width = 5

    input_data = {
        "input": np.random.rand(batch_size, input_height, input_width, input_channels).astype(np.float32),
        "weight": np.random.rand(kernel_size, kernel_size, input_channels, output_channels).astype(np.float32),
        "bias": np.random.rand(output_channels).astype(np.float32),
        "stride": (2, 2),
        "padding": (1, 1),
        "output_padding": (1, 1),
        "groups": 1,
        "dilation": (1, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()