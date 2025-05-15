import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

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

    if not cpu:
        input_tensor = input_tensor.cuda()

    conv3d = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                             padding_mode=padding_mode)

    if not cpu:
        conv3d = conv3d.cuda()
    
    result = conv3d(input_tensor)

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
        input_np = input_dict["input"]
        input_tensor = tf.constant(input_np)
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if isinstance(stride, int):
            stride_d = stride_h = stride_w = stride
        else:
            stride_d, stride_h, stride_w = stride
        
        if isinstance(dilation, int):
            dilation_d = dilation_h = dilation_w = dilation
        else:
            dilation_d, dilation_h, dilation_w = dilation

        if isinstance(kernel_size, int):
            kernel_size_d = kernel_size_h = kernel_size_w = kernel_size
        else:
            kernel_size_d, kernel_size_h, kernel_size_w = kernel_size

        if input_tensor.ndim == 5:
            N, Cin, Din, Hin, Win = input_np.shape
        elif input_tensor.ndim == 4:
            Cin, Din, Hin, Win = input_np.shape
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            N = 1
        else:
            raise ValueError("Input tensor must have 4 or 5 dimensions")

        Dout = int((Din + 2 * padding[0] - dilation * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        Hout = int((Hin + 2 * padding[1] - dilation * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        Wout = int((Win + 2 * padding[2] - dilation * (kernel_size[2] - 1) - 1) / stride[2] + 1)
        
        kernel_shape = (kernel_size_d, kernel_size_h, kernel_size_w, in_channels // groups, out_channels)
        kernel = tf.random.normal(kernel_shape)

        if bias:
            bias_val = tf.random.normal([out_channels])
        else:
            bias_val = None

        if padding_mode == 'zeros':
            if isinstance(padding, int):
                padding_d = padding_h = padding_w = padding
            else:
                padding_d, padding_h, padding_w = padding

            padding_tf = [[0, 0], [padding_d, padding_d], [padding_h, padding_h], [padding_w, padding_w], [0,0]]
            input_tensor_padded = tf.pad(input_tensor, padding_tf, "CONSTANT")
            padding_type = 'VALID'
        elif padding_mode == 'same':
            padding_type = 'SAME'
            input_tensor_padded = input_tensor
        else:
            raise ValueError(f"Padding mode {padding_mode} is not supported in tensorflow version")

        result = tf.nn.conv3d(input=input_tensor_padded, filters=kernel, strides=[1, stride_d, stride_h, stride_w, 1], padding=padding_type, dilations=[1, dilation_d, dilation_h, dilation_w, 1])

        if bias_val is not None:
            result = tf.nn.bias_add(result, bias_val)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(20, 16, 10, 50, 16).astype(np.float32),
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": (3, 5, 2),
        "stride": (2, 1, 1),
        "padding": (4, 2, 0),
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_output_shape = torch_result["result"].shape
    tf_output_shape = tf_result["result"].shape

    min_C = min(torch_output_shape[1], tf_output_shape[1])
    min_D = min(torch_output_shape[2], tf_output_shape[2])
    min_H = min(torch_output_shape[3], tf_output_shape[3])
    min_W = min(torch_output_shape[4], tf_output_shape[4])
    
    torch_cropped = torch_result["result"][:,:min_C,:min_D,:min_H,:min_W]
    tf_cropped = tf_result["result"][:,:min_C,:min_D,:min_H,:min_W]

    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()