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

    conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    
    with torch.no_grad():
        result = conv1d(input_tensor)

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
            raise NotImplementedError("Tensorflow only supports padding_mode='zeros'")

        if dilation != 1:
            raise NotImplementedError("Tensorflow does not directly support dilation in conv1d.  Requires manual implementation")
        
        if groups != 1:
            raise NotImplementedError("Tensorflow requires tf.nn.convolution for groups != 1 which is complex")

        N, Cin, Lin = input_dict["input"].shape

        kernel_shape = (kernel_size, in_channels, out_channels)
        kernel = np.random.randn(*kernel_shape).astype(np.float32)

        kernel_tensor = tf.constant(kernel)
        
        if isinstance(padding, int):
            padding_tf = 'VALID' if padding == 0 else 'SAME'

        elif isinstance(padding, tuple):
            if padding[0] == 0 and padding[1] == 0:
                padding_tf = "VALID"
            else:
                padding_tf = "SAME"
                
            padding = padding[0]

        else:
            padding_tf = 'VALID'

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1])
        result = tf.nn.conv1d(input_tensor, filters=kernel_tensor, stride=stride, padding=padding_tf)

        if bias:
            bias_val = np.random.randn(out_channels).astype(np.float32)
            bias_tensor = tf.constant(bias_val)
            result = tf.nn.bias_add(result, bias_tensor)

        result = tf.transpose(result, perm=[0, 2, 1]).numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(2, 16, 50).astype(np.float32),
        "in_channels": 16,
        "out_channels": 33,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": "zeros"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()