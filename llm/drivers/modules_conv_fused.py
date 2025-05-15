import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    from torch.nn.intrinsic.qat.modules.conv_fused import ConvBn1d, ConvBn2d, ConvBn3d, ConvReLU1d, ConvReLU2d, ConvReLU3d, ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d
    from torch.quantization import QConfig
    from torch.quantization import default_observer

    conv = torch.tensor(input_dict["conv"])
    bn = torch.tensor(input_dict["bn"])
    relu = input_dict.get("relu", False)
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        conv = conv.cuda()
        bn = bn.cuda()
        input_tensor = input_tensor.cuda()

    qconfig = QConfig(activation=default_observer.Observer.with_args(dtype=torch.qint8), weight=default_observer.default_weight_observer)

    if len(input_dict["input"].shape) == 2:
        in_channels = input_dict["input"].shape[1]
        m = ConvBn1d(in_channels, num_features, kernel_size=1)
        m.qconfig = qconfig
        m.weight.data = conv
    elif len(input_dict["input"].shape) == 3:
        in_channels = input_dict["input"].shape[1]
        kernel_size = conv.shape[2]
        m = ConvBn2d(in_channels, num_features, kernel_size=kernel_size)
        m.qconfig = qconfig
        m.weight.data = conv
    else:
        in_channels = input_dict["input"].shape[1]
        kernel_size = conv.shape[2]
        m = ConvBn3d(in_channels, num_features, kernel_size=kernel_size)
        m.qconfig = qconfig
        m.weight.data = conv

    m.bias.data = torch.zeros(num_features)
    
    result = m(input_tensor)
    
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
        conv = tf.constant(input_dict["conv"])
        bn = tf.constant(input_dict["bn"])
        relu = input_dict.get("relu", False)
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        input_tensor = tf.constant(input_dict["input"])

        conv_w = tf.transpose(tf.reshape(conv, [num_features, -1]), perm=[1, 0])
        conv_b = bn[num_features*2:num_features*3]

        mean = bn[0:num_features]
        variance = bn[num_features:num_features*2]
        gamma = bn[num_features*3:num_features*4]
        beta = bn[num_features*4:]

        scale = gamma / tf.math.sqrt(variance + eps)
        bias = beta - mean * scale
        
        W_hat = conv_w * scale
        b_hat = conv_b * scale + bias

        if len(input_dict["input"].shape) == 2:
            in_channels = input_dict["input"].shape[1]
            input_expanded = tf.expand_dims(input_tensor, axis=0)
            W_hat_reshaped = tf.expand_dims(W_hat, axis=0)
            conv_result = tf.nn.conv1d(input_expanded, W_hat_reshaped, stride=1, padding="VALID")
            conv_result = tf.squeeze(conv_result, axis=0)
            result = conv_result + b_hat
        elif len(input_dict["input"].shape) == 3:
            in_channels = input_dict["input"].shape[1]
            kernel_size = conv.shape[2]
            input_expanded = tf.expand_dims(input_tensor, axis=0)
            W_hat_reshaped = tf.transpose(conv, perm=[2,3,1,0])
            conv_result = tf.nn.conv2d(input_expanded, W_hat_reshaped, strides=[1,1,1,1], padding="VALID")
            conv_result = tf.squeeze(conv_result, axis=0)
            result = conv_result + b_hat
        else:
            in_channels = input_dict["input"].shape[1]
            kernel_size = conv.shape[2]
            input_expanded = tf.expand_dims(input_tensor, axis=0)
            W_hat_reshaped = tf.transpose(conv, perm=[2,3,4,1,0])
            conv_result = tf.nn.conv3d(input_expanded, W_hat_reshaped, strides=[1,1,1,1,1], padding="VALID")
            conv_result = tf.squeeze(conv_result, axis=0)
            result = conv_result + b_hat
        
        if relu:
            result = tf.nn.relu(result)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_shape = (1, 3, 3)
    num_features = 2
    kernel_size = 2
    input_channels = 3

    conv = np.random.rand(num_features, input_channels, kernel_size, kernel_size).astype(np.float32)
    bn = np.random.rand(num_features*5).astype(np.float32)
    input_tensor = np.random.rand(*input_shape).astype(np.float32)

    input_data = {
        "conv": conv,
        "bn": bn,
        "num_features": num_features,
        "input": input_tensor,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()