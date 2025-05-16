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
    padding = input_dict.get("padding", (0,))
    stride = input_dict.get("stride", (1,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    weight = weight.unsqueeze(0).unsqueeze(0)

    if bias is not None:
        if not cpu:
            result = torch.nn.functional.relu(torch.convolution(input_tensor, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups, transposed=False, output_padding=(0,)))
        else:
            result = torch.nn.functional.relu(torch.convolution(input_tensor, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups, transposed=False, output_padding=(0,)))
    else:
        if not cpu:
            result = torch.nn.functional.relu(torch.convolution(input_tensor, weight, None, stride=stride, padding=padding, dilation=dilation, groups=groups, transposed=False, output_padding=(0,)))
        else:
            result = torch.nn.functional.relu(torch.convolution(input_tensor, weight, None, stride=stride, padding=padding, dilation=dilation, groups=groups, transposed=False, output_padding=(0,)))

    if not cpu:
        result = result.cpu()

    result = result.squeeze()
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
        padding = input_dict.get("padding", (0,))
        stride = input_dict.get("stride", (1,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)
        
        padding_tf = 'VALID'
        if padding == (0,):
            padding_tf = 'VALID'
        elif padding == (1,):
            padding_tf = 'SAME'
        else:
            raise ValueError("Only padding=0 and padding=1 are supported")
            
        if dilation != (1,):
            raise ValueError("Dilation is not supported")
        
        if groups != 1:
            raise ValueError("Groups is not supported")
            
        input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], 1])
        weight = tf.reshape(weight, [weight.shape[0], weight.shape[1], 1, 1])
        
        conv = tf.nn.conv2d(input_tensor, weight, strides=[1, stride[0], stride[0], 1], padding=padding_tf)

        if bias is not None:
            conv = tf.nn.bias_add(conv, bias)

        result = tf.nn.relu(conv)
        result = tf.reshape(result, [result.shape[1], result.shape[2]])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "weight": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "padding": (0,),
        "stride": (1,),
        "dilation": (1,),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()