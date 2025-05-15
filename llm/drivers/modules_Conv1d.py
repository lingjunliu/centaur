import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
        bias = torch.tensor(bias)
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

    in_channels = 1
    out_channels = weight.shape[0]
    kernel_size = weight.shape[1]

    conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias is not None)
    conv1d.weight.data = weight.unsqueeze(1)
    if bias is not None:
        conv1d.bias.data = bias

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    result = conv1d(input_tensor).squeeze(0).squeeze(0)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias, dtype=tf.float32)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=2)

        weight = tf.expand_dims(weight, axis=1)

        if padding_mode == 'zeros':
            padding_tf = 'VALID' if padding == 0 else 'SAME'
            if padding > 0:
                pad_total = (weight.shape[2] - 1) * dilation
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                input_tensor = tf.pad(input_tensor, [[0, 0], [pad_beg, pad_end], [0, 0]])
            
            result = tf.nn.conv1d(input_tensor, weight, stride=stride, padding=padding_tf)

        else:
            raise ValueError(f"Padding mode {padding_mode} not implemented in tensorflow")
        
        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = tf.squeeze(result, axis=[0,2])
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "weight": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "weight": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()