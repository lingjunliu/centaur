import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel = torch.tensor(input_dict["weight"])
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    dilation = input_dict.get("dilation", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        kernel = kernel.cuda()
    
    result = torch.nn.functional.conv_transpose3d(
        input_tensor,
        kernel,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    
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
        kernel = tf.constant(input_dict["weight"])
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        groups = input_dict.get("groups", 1)
        dilation = input_dict.get("dilation", 1)

        input_shape = input_tensor.shape.as_list()
        kernel_shape = kernel.shape.as_list()
        
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(padding, int):
            padding = [padding] * 3
        if isinstance(dilation, int):
            dilation = [dilation] * 3
        if isinstance(output_padding, int):
             output_padding = [output_padding] * 3
            
        batch_size = input_shape[0]
        in_channels = input_shape[1]
        in_depth = input_shape[2]
        in_height = input_shape[3]
        in_width = input_shape[4]
        
        out_channels = kernel_shape[1] * groups
        
        kernel_depth = kernel_shape[2]
        kernel_height = kernel_shape[3]
        kernel_width = kernel_shape[4]

        output_shape = [
            batch_size,
            out_channels,
            (in_depth - 1) * stride[0] + output_padding[0] + 1 ,
            (in_height - 1) * stride[1] + output_padding[1] + 1,
            (in_width - 1) * stride[2] + output_padding[2] + 1
        ]
        
        strides = [1, stride[0], stride[1], stride[2], 1]
        
        if padding[0] == 0 and padding[1] == 0 and padding[2] == 0:
          padding_tf = 'VALID'
        else:
          padding_tf = 'SAME'

        result = tf.nn.conv3d_transpose(
            input=input_tensor,
            filters=kernel,
            output_shape=output_shape,
            strides=strides,
            padding=padding_tf,
            data_format="NCDHW",
            dilations=[1, dilation[0], dilation[1], dilation[2], 1]
        )
        if padding_tf == 'SAME':
            pad_depth = padding[0]
            pad_height = padding[1]
            pad_width = padding[2]

            result = result[:, :, pad_depth:-pad_depth, pad_height:-pad_height, pad_width:-pad_width]

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.random.rand(1, 3, 4, 5, 6).astype(np.float32),
        "weight": np.random.rand(3, 3, 3, 3, 3).astype(np.float32),
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 1,
        "dilation": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()