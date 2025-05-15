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

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    weight = weight.unsqueeze(1).unsqueeze(1)
    
    result = torch.convolution(input_tensor, weight, bias, stride=(stride,) if isinstance(stride, int) else stride, padding=(padding,) if isinstance(padding, int) else padding, dilation=(dilation,) if isinstance(dilation, int) else dilation, transposed=False, output_padding=(0,) if isinstance(0, int) else 0, groups=groups)

    if not cpu:
        result = result.cpu()
    
    return {"result": result[0,0].numpy()}

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
        
        input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], 1, 1])
        weight = tf.reshape(weight, [weight.shape[0], 1, 1, 1])

        if isinstance(stride, int):
            strides = [1, stride, 1, 1]
        else:
            strides = [1, stride[0], 1, 1]

        if isinstance(dilation, int):
            dilations = [1, dilation, 1, 1]
        else:
            dilations = [1, dilation[0], 1, 1]

        if isinstance(padding, int):
            padding_val = 'VALID' if padding == 0 else 'SAME'
        else:
            if padding[0] == 0 and padding[1] == 0:
                padding_val = 'VALID'
            else:
                padding_val = 'SAME'

        result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_val, dilations=dilations)
        
        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = tf.reshape(result, [result.shape[1]])
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "weight": np.array([[0.1], [0.2]], dtype=np.float32),
        "bias": np.array([0.1], dtype=np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()