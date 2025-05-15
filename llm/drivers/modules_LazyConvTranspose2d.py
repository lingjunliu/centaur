import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    out_channels = input_dict["out_channels"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_conv_transpose2d = torch.nn.LazyConvTranspose2d(
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    if not cpu:
        lazy_conv_transpose2d = lazy_conv_transpose2d.cuda()

    result = lazy_conv_transpose2d(input_tensor)

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
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        out_channels = input_dict["out_channels"]

        input_shape = input_tensor.shape
        in_channels = input_shape[-1]
        
        kernel_shape = [kernel_size, kernel_size, out_channels, in_channels]
        
        kernel = tf.random.normal(kernel_shape)

        if bias:
            bias_tensor = tf.Variable(tf.zeros(shape=(out_channels,)))
        else:
            bias_tensor = None

        batch_size = 1
        input_tensor_expanded = tf.reshape(input_tensor, [batch_size, input_shape[0], input_shape[1], input_shape[2]])
        output_shape_tensor = [batch_size,
                               (input_shape[0] - 1) * stride + kernel_size + (stride-1)*(input_shape[0]-1) - 2*padding + output_padding ,
                               (input_shape[1] - 1) * stride + kernel_size + (stride-1)*(input_shape[1]-1) - 2*padding + output_padding,
                               out_channels]

        result = tf.nn.conv2d_transpose(
            input_tensor_expanded,
            kernel,
            output_shape=tf.stack(output_shape_tensor),
            strides=[1, stride, stride, 1],
            padding="VALID",
            data_format="NHWC",
            dilations=[1, dilation, dilation, 1]
        )
        
        if padding > 0:
            result = tf.pad(result, [[0, 0], [padding, padding], [padding, padding], [0, 0]])

        if bias_tensor is not None:
            result = tf.nn.bias_add(result, bias_tensor, data_format="NHWC")

        result = tf.squeeze(result, axis=0)
        result = result.numpy()

        return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 2, 3).astype(np.float32),
        "out_channels": 6,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
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