import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.quantized as nniq

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1, 1, 1))
    padding = input_dict.get("padding", (0, 0, 0))
    dilation = input_dict.get("dilation", (1, 1, 1))
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    conv_relu_3d = nniq.ConvReLU3d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"], stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, bias=(bias is not None))

    with torch.no_grad():
        conv_relu_3d.weight.copy_(weight)
        if bias is not None:
            conv_relu_3d.bias.copy_(bias)

    result = conv_relu_3d(input_tensor)
    
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
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None
        stride = input_dict.get("stride", (1, 1, 1))
        padding = input_dict.get("padding", (0, 0, 0))
        dilation = input_dict.get("dilation", (1, 1, 1))
        groups = input_dict.get("groups", 1)
        padding_mode = 'VALID'
        if padding != (0,0,0):
            padding_mode = 'SAME'

        if groups != 1:
          raise ValueError("Groups != 1 is not supported in tensorflow")

        strides = [1] + list(stride) + [1]
        dilations = [1] + list(dilation) + [1]
        
        weight = tf.transpose(weight, perm=[2, 3, 4, 1, 0])

        result_conv = tf.nn.conv3d(input_tensor[None,...], weight, strides=strides, padding=padding_mode, dilations=dilations)[0,...]

        if bias is not None:
            result_conv = tf.nn.bias_add(result_conv, bias)

        result = tf.nn.relu(result_conv)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "groups": 1,
        "padding_mode": 'zeros'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()