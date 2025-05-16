import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    in_channels = input_dict.get("in_channels", 0)
    out_channels = input_dict.get("out_channels", 0)
    kernel_size = input_dict.get("kernel_size", 1)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    lazy_conv1d = nn.LazyConv1d(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    if not cpu:
      lazy_conv1d = lazy_conv1d.cuda()

    result = lazy_conv1d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    in_channels = input_dict.get("in_channels", 0)
    out_channels = input_dict.get("out_channels", 0)
    kernel_size = input_dict.get("kernel_size", 1)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    
    if len(input_tensor.shape) == 1:
        input_tensor = tf.reshape(input_tensor, (1, 1, -1))

    if out_channels == 0:
        out_channels = 1

    kernel = tf.random.normal((kernel_size, input_tensor.shape[1], out_channels), dtype=tf.float32)

    if padding > 0:
        input_tensor = tf.pad(input_tensor, [[0, 0], [0, 0], [padding, padding]], "CONSTANT")

    input_tensor = tf.expand_dims(input_tensor, axis=0)
    
    kernel = tf.reshape(kernel, (kernel_size, 1, input_tensor.shape[2], out_channels))

    result = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride, 1, 1], padding="VALID")

    if bias:
        bias_term = tf.random.normal((out_channels,), dtype=tf.float32)
        result = tf.nn.bias_add(result, bias_term)
    
    result = tf.squeeze(result, axis=0)
    result = result[:,0,:]
    result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5).astype(np.float32),
        "kernel_size": 2,
        "out_channels": 1,
        "stride": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()