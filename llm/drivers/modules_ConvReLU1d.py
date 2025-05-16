import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.intrinsic.quantized.modules import ConvReLU1d

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

    conv_relu = ConvReLU1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    with torch.no_grad():
        conv_relu.weight.data.copy_(torch.tensor(input_dict["weight"]))
        if bias:
            conv_relu.bias.data.copy_(torch.tensor(input_dict["bias_value"]))

    if not cpu:
        conv_relu = conv_relu.cuda()
    
    result = conv_relu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'CONSTANT')

        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias_value = tf.constant(input_dict["bias_value"], dtype=tf.float32) if bias else None

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        weight = tf.transpose(tf.constant(input_dict["weight"], dtype=tf.float32), perm=[1, 0])
        weight = tf.expand_dims(weight, axis=0)

        if padding > 0:
            input_tensor = tf.pad(input_tensor, [[0, 0], [0, 0], [padding, padding]], mode=padding_mode)
        
        result = tf.nn.conv1d(input_tensor, filters=weight, stride=stride, padding='VALID', dilation_rate=dilation)
        
        if bias:
            result = tf.nn.bias_add(result, bias_value)

        result = tf.nn.relu(result)

        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=0)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 3,
        "weight": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "bias_value": np.array([0.5], dtype=np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()