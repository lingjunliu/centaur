import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn import LazyConv2d

    input_tensor = torch.tensor(input_dict["input"]).permute(0, 3, 1, 2)
    in_channels = input_dict.get("in_channels", 'None')
    out_channels = input_dict.get("out_channels", 16)
    kernel_size = input_dict.get("kernel_size", 3)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()

    if in_channels == 'None':
        module = LazyConv2d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
    else:
        in_channels = int(in_channels)
        module = torch.nn.Conv2d(
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
        
        if not cpu:
            module = module.cuda()

        module.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]).permute(3, 2, 0, 1))
        if bias:
            module.bias = torch.nn.Parameter(torch.tensor(input_dict["bias_val"]))
    

    result = module(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.keras.layers import Conv2D

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        in_channels = input_dict.get("in_channels", 'None')
        out_channels = input_dict.get("out_channels", 16)
        kernel_size = input_dict.get("kernel_size", 3)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        if in_channels == 'None':
            
            in_channels = int(input_tensor.shape[-1])

            if padding_mode != 'zeros':
                raise NotImplementedError("Tensorflow does not support padding modes other than zeros")
                
            module = Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=stride,
                padding='valid' if padding == 0 else 'same',
                dilation_rate=dilation,
                groups=groups,
                use_bias=bias,
                kernel_initializer=tf.constant_initializer(np.zeros((kernel_size, kernel_size, in_channels, out_channels))),
                bias_initializer=tf.constant_initializer(np.zeros(out_channels)),
                data_format='channels_last'
            )

            result = module(input_tensor)
        else:
            in_channels = int(in_channels)
            if padding_mode != 'zeros':
                raise NotImplementedError("Tensorflow does not support padding modes other than zeros")

            module = Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=stride,
                padding='valid' if padding == 0 else 'same',
                dilation_rate=dilation,
                groups=groups,
                use_bias=bias,
                kernel_initializer=tf.constant_initializer(np.transpose(input_dict["weight"], (2, 3, 0, 1))),
                bias_initializer=tf.constant_initializer(input_dict["bias_val"]) if bias else 'zeros',
                data_format='channels_last'
            )
            result = module(input_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 5, 5, 3).astype(np.float32),
        "in_channels": 'None',
        "out_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "bias": False,
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    tf_result["result"] = np.transpose(tf_result["result"], (0, 3, 1, 2))
    torch_result["result"] = np.transpose(torch_result["result"], (0, 2, 3, 1))

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 5, 5, 3).astype(np.float32),
        "in_channels": 3,
        "out_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "bias": True,
        "weight": np.random.rand(3, 3, 3, 2).astype(np.float32),
        "bias_val": np.random.rand(2).astype(np.float32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    tf_result["result"] = np.transpose(tf_result["result"], (0, 3, 1, 2))
    torch_result["result"] = np.transpose(torch_result["result"], (0, 2, 3, 1))
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()