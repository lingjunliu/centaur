import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = torch.tensor(input_dict["weight"])
    bias_tensor = torch.tensor(input_dict["bias"])
    stride = input_dict.get("stride", (1, 1))
    padding = input_dict.get("padding", (0, 0))
    dilation = input_dict.get("dilation", (1, 1))
    groups = input_dict.get("groups", 1)
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        bias_tensor = bias_tensor.cuda()

    conv = F.conv2d(input_tensor, weight_tensor, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    result = torch.relu(conv + alpha * bias_tensor.view(1, -1, 1, 1))


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
        input_tensor = tf.constant(input_dict["input"])
        weight_tensor = tf.constant(input_dict["weight"])
        bias_tensor = tf.constant(input_dict["bias"])
        stride = input_dict.get("stride", (1, 1))
        padding = input_dict.get("padding", (0, 0))
        dilation = input_dict.get("dilation", (1, 1))
        groups = input_dict.get("groups", 1)
        alpha = input_dict.get("alpha", 1.0)

        if len(stride) == 1:
            strides = [1, stride[0], stride[0], 1]
        elif len(stride) == 2:
            strides = [1, stride[0], stride[1], 1]
        else:
            raise ValueError("Stride must be a tuple of length 1 or 2")

        if len(padding) == 1:
            padding_tf = 'SAME'
        elif len(padding) == 2 and padding[0] == 0 and padding[1] == 0:
            padding_tf = 'VALID'
        else:
            padding_list = [ [0,0], [padding[0], padding[0]], [padding[1], padding[1]], [0,0] ]
            input_tensor = tf.pad(input_tensor, padding_list)
            padding_tf = 'VALID'

        if len(dilation) == 1:
            rates = [1, dilation[0], dilation[0], 1]
        elif len(dilation) == 2:
            rates = [1, dilation[0], dilation[1], 1]
        else:
            raise ValueError("Dilation must be a tuple of length 1 or 2")

        input_shape = input_tensor.shape
        weight_shape = weight_tensor.shape

        in_channels = input_shape[-1]
        kernel_in_channels = weight_shape[1]

        if in_channels % kernel_in_channels != 0 and groups > 1:
            raise ValueError("Input channels must be divisible by kernel in channels when groups > 1")

        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        if len(weight_shape) == 3:
            weight_tensor = tf.expand_dims(weight_tensor, axis=0)
            
        if groups == 1:
            conv = tf.nn.conv2d(input_tensor, weight_tensor, strides=strides, padding=padding_tf, dilations=rates)
        else:
            weight_groups = tf.split(weight_tensor, num_or_size_splits=groups, axis=0)
            input_groups = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
            conv_groups = []
            for i in range(groups):
                conv_groups.append(tf.nn.conv2d(input_groups[i], weight_groups[i], strides=strides, padding=padding_tf, dilations=rates))
            conv = tf.concat(conv_groups, axis=3)


        bias_tensor = tf.reshape(bias_tensor, [1, 1, 1, -1])
        added = tf.nn.bias_add(conv, bias_tensor, data_format='NHWC')
        result = tf.nn.relu(added)

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 32, 32, 3).astype(np.float32),
        "weight": np.random.rand(5, 5, 3, 6).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()