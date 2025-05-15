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

    conv = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode
    )
    relu = torch.nn.ReLU()

    if not cpu:
        conv = conv.cuda()
        relu = relu.cuda()

    input_tensor = input_tensor.unsqueeze(0).transpose(1, 2)
    result = conv(input_tensor)
    result = relu(result)

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
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if padding_mode != 'zeros':
            raise ValueError("Tensorflow version only supports padding_mode='zeros'")

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1])

        kernel = tf.random.normal((kernel_size, in_channels // groups, out_channels))

        if bias:
            b = tf.random.normal((out_channels,))
        else:
            b = tf.zeros((out_channels,))

        conv = tf.nn.conv1d(
            input_tensor,
            kernel,
            stride=stride,
            padding='VALID',
            data_format='NWC'
        )
        
        if bias:
            conv = tf.nn.bias_add(conv, b, data_format='NWC')

        if padding > 0:
             conv = tf.pad(conv, [[0, 0], [padding, padding], [0, 0]])
            
        relu = tf.nn.relu(conv)

        result = relu.numpy()
        result = np.squeeze(result, axis=0)
        result = np.transpose(result, axes=[1,0])
        

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 3).astype(np.float32),
        "in_channels": 3,
        "out_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()