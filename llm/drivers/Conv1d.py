import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.nn.functional.conv1d(input_tensor, weight, bias, stride, padding, dilation, groups)

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
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'CONSTANT') 

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1])
        weight = tf.transpose(weight, perm=[2, 1, 0])
        
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        result = tf.nn.conv1d(
            input_tensor,
            filters=weight,
            stride=stride,
            padding='VALID',
            data_format='NWC'
        )
        
        if padding > 0:
            input_tensor_padded = tf.pad(input_tensor, [[0, 0], [padding, padding], [0, 0]], mode=padding_mode)
            result = tf.nn.conv1d(
                input_tensor_padded,
                filters=weight,
                stride=stride,
                padding='VALID',
                data_format='NWC'
            )


        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format='NWC')

        result = tf.squeeze(result, axis=0)
        
        result = tf.transpose(result, perm=[1, 0])

        result = result.numpy()

        if dilation != 1:
            result_dilated = []
            for i in range(0, result.shape[0], dilation):
                result_dilated.append(result[i])
            result = np.array(result_dilated)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_channels = 3
    kernel_size = 3
    output_channels = 2
    length = 5

    input_data = {
        "input": np.random.rand(1, input_channels, length).astype(np.float32),
        "weight": np.random.rand(output_channels, input_channels, kernel_size).astype(np.float32),
        "bias": np.random.rand(output_channels).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()