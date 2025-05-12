import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1,))
    padding = input_dict.get("padding", (0,))
    output_padding = input_dict.get("output_padding", (0,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
    
    result = torch.cudnn_convolution_transpose(input_tensor, weight, bias, (int(padding[0]), int(padding[1])), (int(stride[0]), int(stride[1])), (int(dilation[0]), int(dilation[1])), groups, (int(output_padding[0]), int(output_padding[1])))
    
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
        stride = input_dict.get("stride", (1,))
        padding = input_dict.get("padding", (0,))
        output_padding = input_dict.get("output_padding", (0,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)

        data_format = 'N'+'H'*(len(stride))+'C'

        input_shape = input_tensor.shape
        kernel_size = weight.shape[2:]
        num_filters = weight.shape[1]
        
        output_shape = (input_shape[0],) + tuple((input_shape[i+1]-1)*stride[i-1] - 2*padding[i-1] + dilation[i-1]*(kernel_size[i-1]-1) + output_padding[i-1] + 1 for i in range(1, len(input_shape)-1)) + (num_filters,)
        
        result = tf.nn.conv2d_transpose(input_tensor, weight, output_shape, (1,)+stride+(1,), padding='VALID', data_format=data_format)

        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format=data_format)
    
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10).astype(np.float32),
        "weight": np.random.rand(3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "stride": (2, 2),
        "padding": (1, 1),
        "output_padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()