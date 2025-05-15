import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
        bias = torch.tensor(bias)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.conv1d(input_tensor.unsqueeze(0).unsqueeze(0), weight.unsqueeze(0), bias, stride, padding, dilation, groups).squeeze()

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
        weight = tf.constant(input_dict["weight"])
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias)
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=2)
        weight = tf.expand_dims(weight, axis=0)
        weight = tf.transpose(weight, perm=[2, 1, 0])

        padding_enum = 'VALID'
        if padding != 0:
            padding_enum = 'SAME'

        result = tf.nn.conv1d(input_tensor, filters=weight, stride=stride, padding=padding_enum, data_format='NWC')

        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format='NWC')

        result = tf.squeeze(result, axis=[0,2])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "weight": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "padding": 0,
        "stride": 1,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()