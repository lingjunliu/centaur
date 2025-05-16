import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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

    conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    if 'weight' in input_dict:
        conv2d.weight = torch.nn.Parameter(torch.tensor(input_dict['weight']))
    if 'bias_val' in input_dict and bias:
        conv2d.bias = torch.nn.Parameter(torch.tensor(input_dict['bias_val']))
    elif not bias:
        conv2d.bias = None
    
    if not cpu:
        conv2d = conv2d.cuda()
    
    result = conv2d(input_tensor)

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
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("TensorFlow version only supports padding_mode='zeros'")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(padding, int):
            padding = (padding, padding)

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])
            
        weight_shape = (kernel_size[0], kernel_size[1], in_channels // groups, out_channels)
        weight = tf.constant(input_dict['weight']) if 'weight' in input_dict else tf.random.normal(shape=weight_shape)


        if bias:
            bias_val = tf.constant(input_dict['bias_val']) if 'bias_val' in input_dict else tf.random.normal(shape=(out_channels,))
        else:
            bias_val = None


        if groups == 1:
            strides = [1, stride[0], stride[1], 1]
            rates = [1, dilation[0], dilation[1], 1]
            
            if padding == 0:
                padding_tf = 'VALID'
            else:
                padding_tf = 'SAME'

            result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_tf, dilations=rates)

            if bias:
                result = tf.nn.bias_add(result, bias_val)
        else:
            raise ValueError("TensorFlow version does not support groups != 1")

        result = tf.transpose(result, perm=[0, 3, 1, 2])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "weight": np.random.rand(8, 3, 3, 3).astype(np.float32),
        "bias_val": np.random.rand(8).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    input_data['weight'] = np.transpose(input_data['weight'], (2, 3, 1, 0))
    tf_result = tensorflow_version(input_data)

    th, tw = torch_result['result'].shape[2:]
    tfh, tfw = tf_result['result'].shape[2:]
    h_diff = th - tfh
    w_diff = tw - tfw

    if h_diff > 0 or w_diff > 0:
        pad_top = 0
        pad_bottom = h_diff
        pad_left = 0
        pad_right = w_diff

        padding = [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]]
        tf_result['result'] = np.pad(tf_result['result'], padding, mode='constant')


    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "weight": np.random.rand(8, 3, 3, 3).astype(np.float32),
        "bias_val": np.random.rand(8).astype(np.float32)
    }
    torch_result = torch_version(input_data)
    input_data['weight'] = np.transpose(input_data['weight'], (2, 3, 1, 0))
    tf_result = tensorflow_version(input_data)
    th, tw = torch_result['result'].shape[2:]
    tfh, tfw = tf_result['result'].shape[2:]
    h_diff = th - tfh
    w_diff = tw - tfw

    if h_diff > 0 or w_diff > 0:
        pad_top = 0
        pad_bottom = h_diff
        pad_left = 0
        pad_right = w_diff

        padding = [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]]
        tf_result['result'] = np.pad(tf_result['result'], padding, mode='constant')

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()