import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.intrinsic.quantized as nniq

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    conv_relu = nniq.ConvReLU2d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"], stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

    with torch.no_grad():
        conv_relu.weight[:] = weight
        if bias is not None:
            conv_relu.bias[:] = bias
    
    result = conv_relu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'VALID') 
        
        if padding != 0:
          input_tensor = tf.pad(input_tensor, [[0,0],[padding,padding],[padding,padding],[0,0]])

        if padding_mode != "zeros" and padding != 0:
          raise ValueError("Tensorflow only supports zero padding with Conv2D.")

        if groups > 1:
            kernel_height, kernel_width, in_channels_per_group, out_channels_per_group = weight.shape[0], weight.shape[1], input_dict["in_channels"] // groups, input_dict["out_channels"] // groups
            input_slices = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
            weight_slices = tf.split(weight, num_or_size_splits=groups, axis=3)
            
            outputs = []
            for i in range(groups):
                conv = tf.nn.conv2d(input_slices[i], weight_slices[i], strides=[1, stride, stride, 1], padding=padding_mode, dilations=[1, dilation, dilation, 1])
                if bias is not None:
                    conv = tf.nn.bias_add(conv, bias[i*out_channels_per_group:(i+1)*out_channels_per_group])
                outputs.append(conv)
            
            result = tf.concat(outputs, axis=3)
        else:
          result = tf.nn.conv2d(input_tensor, weight, strides=[1, stride, stride, 1], padding=padding_mode, dilations=[1, dilation, dilation, 1])
          if bias is not None:
              result = tf.nn.bias_add(result, bias)

        result = tf.nn.relu(result)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 32, 32, 3).astype(np.float32),
        "weight": np.random.rand(16, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": (3, 3),
        "padding": 1,
        "stride": 1,
        "dilation": 1,
        "groups": 1,
        "padding_mode": 'zeros'
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()