import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict.get("bias", np.zeros(input_dict["input"].shape[1]))) if "bias" in input_dict else None
    stride = tuple(map(int, input_dict.get("stride", (1,1))))
    padding = tuple(map(int, input_dict.get("padding", (0,0))))
    output_padding = tuple(map(int, input_dict.get("output_padding", (0,0))))
    groups = input_dict.get("groups", 1)
    dilation = input_dict.get("dilation", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.cudnn_convolution_transpose(input_tensor, weight, bias, padding, output_padding, stride, dilation, groups)

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
        bias = tf.constant(input_dict.get("bias", np.zeros(input_dict["input"].shape[1]))) if "bias" in input_dict else None
        stride = tuple(map(int, input_dict.get("stride", (1,1))))
        padding = tuple(map(int, input_dict.get("padding", (0,0))))
        output_padding = tuple(map(int, input_dict.get("output_padding", (0,0))))
        groups = input_dict.get("groups", 1)
        dilation = input_dict.get("dilation", 1)

        input_shape = input_dict["input"].shape
        kernel_shape = input_dict["weight"].shape
        
        batch_size = input_shape[0]
        in_channels = input_shape[1]
        in_height = input_shape[2]
        in_width = input_shape[3]
        
        out_channels = kernel_shape[1] * groups
        
        kernel_height = kernel_shape[2]
        kernel_width = kernel_shape[3]

        stride_height, stride_width = stride
        padding_height, padding_width = padding
        output_padding_height, output_padding_width = output_padding

        out_height = (in_height - 1) * stride_height - 2 * padding_height + dilation * (kernel_height - 1) + output_padding_height + 1
        out_width = (in_width - 1) * stride_width - 2 * padding_width + dilation * (kernel_width - 1) + output_padding_width + 1
        
        output_shape = [batch_size, out_channels, out_height, out_width]
        
        strides = [1, stride_height, stride_width, 1]
        padding_tf = 'VALID' if padding == (0,0) else 'SAME'

        if groups == 1:
          result = tf.nn.conv2d_transpose(input_tensor, weight, output_shape, strides=strides, padding=padding_tf, data_format='NCHW')
          if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format='NCHW')

        else:
          input_groups = tf.split(input_tensor, num_or_size_splits=groups, axis=1)
          weight_groups = tf.split(weight, num_or_size_splits=groups, axis=0)
          
          output_groups = []
          for i in range(groups):
            conv = tf.nn.conv2d_transpose(input_groups[i], weight_groups[i], output_shape, strides=strides, padding=padding_tf, data_format='NCHW')
            output_groups.append(conv)

          result = tf.concat(output_groups, axis=1)
          if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format='NCHW')
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 5, 5).astype(np.float32),
        "weight": np.random.rand(3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "stride": (1,1),
        "padding": (0,0),
        "output_padding": (0,0),
        "groups": 1,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 3, 5, 5).astype(np.float32),
        "weight": np.random.rand(3, 3, 3, 3).astype(np.float32),
        "stride": (2,2),
        "padding": (1,1),
        "output_padding": (1,1),
        "groups": 1,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()