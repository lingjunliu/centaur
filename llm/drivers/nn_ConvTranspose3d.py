import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    dilation = input_dict.get("dilation", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)

    if not cpu:
        m = m.cuda()
    
    result = m(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    dilation = input_dict.get("dilation", 1)
    input_tensor = tf.constant(input_dict["input"])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding, padding)
    
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding)

    input_shape = input_dict["input"].shape
    
    if len(input_shape) == 5:
      batch_size = input_shape[0]
      depth_in = input_shape[2]
      height_in = input_shape[3]
      width_in = input_shape[4]
      
    else:
      raise ValueError("Expected input tensor to have rank 5")

    depth_out = (depth_in - 1) * stride[0] - 2 * padding[0] + dilation * (kernel_size[0] - 1) + output_padding[0] + 1
    height_out = (height_in - 1) * stride[1] - 2 * padding[1] + dilation * (kernel_size[1] - 1) + output_padding[1] + 1
    width_out = (width_in - 1) * stride[2] - 2 * padding[2] + dilation * (kernel_size[2] - 1) + output_padding[2] + 1

    output_shape = (batch_size, depth_out, height_out, width_out, out_channels)

    kernel_shape = (kernel_size[0], kernel_size[1], kernel_size[2],  in_channels, out_channels)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        kernel = tf.Variable(tf.random.normal(kernel_shape))
        
        result = tf.nn.conv3d_transpose(input_tensor, kernel, output_shape=output_shape, strides=(1, stride[0], stride[1], stride[2], 1), padding="VALID")

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "in_channels": 16,
        "out_channels": 33,
        "kernel_size": 3,
        "stride": 2,
        "input": np.random.randn(20, 16, 10, 50, 100).astype(np.float32)
    }

    input_data["input"] = np.reshape(input_data["input"], (20,16,10,50,100))

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data2 = {
        "in_channels": 16,
        "out_channels": 33,
        "kernel_size": (3, 5, 2),
        "stride": (2, 1, 1),
        "padding": (0, 4, 2),
        "input": np.random.randn(20, 16, 10, 50, 100).astype(np.float32)
    }
    input_data2["input"] = np.reshape(input_data2["input"], (20, 16, 10, 50, 100))

    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)
    assert np.allclose(torch_result2["result"], tf_result2["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()