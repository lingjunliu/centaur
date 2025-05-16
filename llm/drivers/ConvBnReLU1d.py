import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig, HistogramObserver, default_weight_observer

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

    qconfig = QConfig(activation=HistogramObserver.with_args(),
                       weight=default_weight_observer)
    conv_bn_relu = nniqat.ConvBnReLU1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig)
    
    if not cpu:
        conv_bn_relu = conv_bn_relu.cuda()
    
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    result = conv_bn_relu(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.squeeze().detach().numpy()}

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
        padding_mode = input_dict.get("padding_mode", 'zeros')
        
        if padding_mode != 'zeros':
            raise ValueError("Tensorflow only supports padding_mode='zeros'")

        input_tensor_expanded = tf.expand_dims(input_tensor, axis=0)
        input_tensor_expanded = tf.expand_dims(input_tensor_expanded, axis=0)
        input_tensor_expanded = tf.transpose(input_tensor_expanded, perm=[0, 2, 1])

        kernel_shape = [kernel_size, in_channels // groups, out_channels]
        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        kernel = tf.Variable(initial_value=kernel_initializer(shape=kernel_shape, dtype='float32'))

        if bias:
            bias_initializer = tf.zeros_initializer()
            bias_var = tf.Variable(initial_value=bias_initializer(shape=[out_channels], dtype='float32'))
        else:
            bias_var = None

        strides = [1, stride]
        padding_tf = 'VALID' if padding == 0 else 'SAME'
        dilations = [1, dilation]

        conv = tf.nn.conv1d(input_tensor_expanded, kernel, stride=strides[1], padding=padding_tf, dilations=dilations[1])

        if bias_var is not None:
            conv = tf.nn.bias_add(conv, bias_var)

        bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)
        bn.build(conv.shape)
        bn_output = bn(conv)
        
        relu = tf.nn.relu(bn_output)
        result = tf.transpose(relu, perm=[0, 2, 1])


        result = tf.squeeze(result, axis=[0]).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 1,
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