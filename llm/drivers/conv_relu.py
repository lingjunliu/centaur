import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.quantized.modules.conv_relu import ConvReLU2d

    input_tensor = torch.tensor(input_dict["input"])
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    in_channels = input_dict["in_channels"]
    out_channels = input_dict["out_channels"]
    kernel_size = input_dict["kernel_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    conv_relu = ConvReLU2d(in_channels, out_channels, kernel_size)

    with torch.no_grad():
        conv_relu.weight.copy_(torch.tensor(input_dict["weight"]))
        conv_relu.bias.copy_(torch.tensor(input_dict["bias"]))
    
    if not cpu:
        conv_relu = conv_relu.cuda()
    
    result = conv_relu(input_tensor)
    
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
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])

        stride = [1, 1]
        padding = 'VALID'
        data_format = 'NHWC'

        conv = tf.nn.conv2d(input_tensor, weight, strides=[1, stride[0], stride[1], 1], padding=padding, data_format=data_format)
        biased = tf.nn.bias_add(conv, bias, data_format=data_format)
        relu = tf.nn.relu(biased)

        result = relu.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 32, 32, 3).astype(np.float32),
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": (3, 3),
        "weight": np.random.rand(16, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()