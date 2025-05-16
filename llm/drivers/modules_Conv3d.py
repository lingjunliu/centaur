import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = None
    if "bias" in input_dict:
        bias = torch.tensor(input_dict["bias"])
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

    result = torch.nn.functional.conv3d(input_tensor, weight, bias, stride, padding, dilation, groups)

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
        weight = tf.constant(input_dict["weight"])
        bias = None
        if "bias" in input_dict:
            bias = tf.constant(input_dict["bias"])
        stride = input_dict.get("stride", (1, 1, 1))
        padding = input_dict.get("padding", (0, 0, 0))
        dilation = input_dict.get("dilation", (1, 1, 1))
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        strides = (1, 1, stride[0], stride[1], stride[2])
        dilations = (1, dilation[0], dilation[1], dilation[2], 1)

        input_tensor_shape = input_tensor.shape
        weight_shape = weight.shape

        if padding_mode == 'zeros':
            if padding == (0,0,0):
                padding_tf = 'VALID'
            else:
                padding_tf = 'VALID'
                pad_depth = padding[0]
                pad_height = padding[1]
                pad_width = padding[2]
                input_tensor = tf.pad(input_tensor, [[0, 0], [pad_depth, pad_depth], [pad_height, pad_height], [pad_width, pad_width], [0, 0]], "CONSTANT")
        else:
            raise NotImplementedError(f"Padding mode {padding_mode} is not supported.")

        if groups == 1:
            #TF doesn't support strides in the batch or depth dimensions
            #input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])
            #weight = tf.transpose(weight, perm=[2, 3, 4, 1, 0])
            result = tf.nn.conv3d(input_tensor, weight, strides=strides, padding=padding_tf, dilations=dilations, data_format='NCDHW')
            #result = tf.transpose(result, perm=[0, 4, 1, 2, 3])
            if bias is not None:
                result = tf.nn.bias_add(result, bias, data_format='NCDHW')
        else:
             raise NotImplementedError("Groups > 1 not implemented")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
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