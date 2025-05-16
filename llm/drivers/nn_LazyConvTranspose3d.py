import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"], requires_grad=False)
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
    
    lazy_conv_transpose_3d = torch.nn.LazyConvTranspose3d(
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        padding_mode=padding_mode
    )
    
    if not cpu:
        lazy_conv_transpose_3d = lazy_conv_transpose_3d.cuda()
        
    result = lazy_conv_transpose_3d(input_tensor)
    
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
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        dilation = input_dict.get("dilation", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        in_channels = int(input_tensor.shape[1])

        kernel_shape = (kernel_size, kernel_size, kernel_size, out_channels, in_channels // groups)
        kernel = tf.random.normal(kernel_shape)

        if bias:
            bias_tensor = tf.random.normal((out_channels,))
        else:
            bias_tensor = None

        if isinstance(stride, int):
            stride = [stride, stride, stride]
        else:
            stride = [stride[0], stride[1], stride[2]]
        
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        else:
            padding = [padding[0], padding[1], padding[2]]

        if isinstance(output_padding, int):
            output_padding = [output_padding, output_padding, output_padding]
        else:
            output_padding = [output_padding[0], output_padding[1], output_padding[2]]
        
        if isinstance(dilation, int):
            dilation = [dilation, dilation, dilation]
        else:
            dilation = [dilation[0], dilation[1], dilation[2]]

        if padding_mode == 'zeros':
            output_shape = (input_tensor.shape[0], out_channels, (input_tensor.shape[2] - 1) * stride[0] + kernel_size - 2 * padding[0] + output_padding[0], (input_tensor.shape[3] - 1) * stride[1] + kernel_size - 2 * padding[1] + output_padding[1], (input_tensor.shape[4] - 1) * stride[2] + kernel_size - 2 * padding[2] + output_padding[2])
            
            result = tf.nn.conv3d_transpose(
                input_tensor,
                kernel,
                output_shape=output_shape,
                strides=[1, stride[0], stride[1], stride[2], 1],
                padding="VALID",
                data_format="NCDHW",
                dilations=[1, dilation[0], dilation[1], dilation[2], 1]
            )
            
            paddings = [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]]]
            result = tf.pad(result, paddings, "CONSTANT")

        else:
            raise NotImplementedError(f"Padding mode {padding_mode} not implemented.")

        if bias_tensor is not None:
            result = tf.nn.bias_add(result, bias_tensor, data_format="NCDHW")
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 4, 5, 6).astype(np.float32),
        "out_channels": 2,
        "kernel_size": 3,
        "stride": (2,2,2),
        "padding": (1,1,1),
        "output_padding": (0,0,0),
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()