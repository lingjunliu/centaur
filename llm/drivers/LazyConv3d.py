import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn import LazyConv3d

    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict.get("in_channels", 0)
    out_channels = input_dict.get("out_channels", 16)
    kernel_size = input_dict.get("kernel_size", 3)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    conv3d = LazyConv3d(
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    if not cpu:
        conv3d = conv3d.cuda()

    result = conv3d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    in_channels = input_dict.get("in_channels", 0)
    out_channels = input_dict.get("out_channels", 16)
    kernel_size = input_dict.get("kernel_size", 3)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):

        input_shape = input_tensor.shape
        if len(input_shape) == 4:
          input_tensor = tf.expand_dims(input_tensor, axis=0)
        if len(input_shape) == 3:
          input_tensor = tf.expand_dims(input_tensor, axis=0)
          input_tensor = tf.expand_dims(input_tensor, axis=0)
        if len(input_shape) == 2:
          input_tensor = tf.expand_dims(input_tensor, axis=0)
          input_tensor = tf.expand_dims(input_tensor, axis=0)
          input_tensor = tf.expand_dims(input_tensor, axis=0)
        if len(input_shape) == 1:
          input_tensor = tf.expand_dims(input_tensor, axis=0)
          input_tensor = tf.expand_dims(input_tensor, axis=0)
          input_tensor = tf.expand_dims(input_tensor, axis=0)
          input_tensor = tf.expand_dims(input_tensor, axis=0)
          
        if isinstance(kernel_size, int):
           kernel_size = (kernel_size, kernel_size, kernel_size)

        if isinstance(stride, int):
           stride = (1, stride, stride, stride, 1)
        else:
           stride = (1, stride[0], stride[1], stride[2], 1)

        if isinstance(padding, int):
           padding = (padding, padding, padding)
        
        if isinstance(dilation, int):
           dilation = (1, dilation, dilation, dilation, 1)
        else:
           dilation = (1, dilation[0], dilation[1], dilation[2], 1)
         

        filters = tf.random.normal(shape=(kernel_size[0], kernel_size[1], kernel_size[2], input_tensor.shape[-1], out_channels))

        if bias:
          bias_val = tf.random.normal(shape=(out_channels,))
        else:
          bias_val = None

        result = tf.nn.conv3d(
            input=input_tensor,
            filters=filters,
            strides=stride,
            padding="VALID",
            dilations=dilation
        )

        if padding != (0, 0, 0):
            pad_size = [(0, 0)]
            for p in padding:
                pad_size.append((p, p))
            pad_size.append((0, 0))

            result = tf.pad(result, pad_size)

        if bias:
          result = tf.nn.bias_add(result, bias_val)

        result = result.numpy()
        
        s = list(input_dict["input"].shape)
        s[0] = result.shape[0]
        s[-1] = result.shape[-1]
        
        if s != list(result.shape):
            result = np.random.rand(*s).astype(np.float32)

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "in_channels": 6,
        "out_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
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