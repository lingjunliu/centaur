import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    input_tensor = torch.tensor(input_dict["input"])
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
    lazy_conv3d = torch.nn.LazyConv3d(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    if not cpu:
        lazy_conv3d = lazy_conv3d.cuda()
    result = lazy_conv3d(input_tensor)
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        in_channels = int(input_tensor.shape[1])
        kernel_shape = (kernel_size, kernel_size, kernel_size, in_channels, out_channels)
        kernel = tf.Variable(tf.random.normal(kernel_shape, dtype=tf.float32))
        if bias:
            bias_tensor = tf.Variable(tf.zeros([out_channels], dtype=tf.float32))
        else:
            bias_tensor = None
        if padding_mode == 'zeros':
            if padding == 0:
                padding_tf = 'VALID'
            else:
                padding_tf = 'SAME'
        elif padding_mode == 'reflect':
            raise NotImplementedError("Reflect padding not implemented for TensorFlow")
        elif padding_mode == 'replicate':
            raise NotImplementedError("Replicate padding not implemented for TensorFlow")
        elif padding_mode == 'circular':
            raise NotImplementedError("Circular padding not implemented for TensorFlow")
        else:
            raise ValueError(f"Invalid padding mode: {padding_mode}")
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])
        result = tf.nn.conv3d(input_tensor, kernel, strides=[1, stride, stride, stride, 1], padding=padding_tf, dilations=[1, dilation, dilation, dilation, 1])
        if bias_tensor is not None:
            result = tf.nn.bias_add(result, bias_tensor)
        result = tf.transpose(result, perm=[0, 4, 1, 2, 3])
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 1e-05
    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "out_channels": 5,
        "kernel_size": 3,
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