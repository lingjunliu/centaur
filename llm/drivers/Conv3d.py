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

    result = torch.nn.functional.conv3d(input_tensor, weight, bias, stride, padding, dilation, groups)
    
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

        if isinstance(stride, int):
            stride = [stride, stride, stride]

        if isinstance(padding, int):
            padding_tf = "VALID" if padding == 0 else "SAME"
        else:
            padding_tf = "VALID"

        if isinstance(dilation, int):
            dilation = [dilation, dilation, dilation]
        
        input_shape = input_tensor.shape
        weight_shape = weight.shape

        if input_shape[2] < weight_shape[2] and padding_tf != "SAME":
            padding_tf = "VALID"
        elif padding != 0 and padding_tf != "VALID":
            padding_tf = "SAME"
        else:
            padding_tf = "VALID"

        if groups == 1:
            result = tf.nn.conv3d(input_tensor, weight, strides=[1, stride[0], stride[1], stride[2], 1], padding=padding_tf, dilations=[1, dilation[0], dilation[1], dilation[2], 1])
        else:
            input_channels = int(input_tensor.shape[-1])
            kernel_channels = int(weight.shape[-1])
            group_size = input_channels // groups

            kernels = tf.split(weight, num_or_size_splits=groups, axis=3)
            inputs = tf.split(input_tensor, num_or_size_splits=groups, axis=4)

            convolved_parts = []
            for i in range(groups):
                convolved_parts.append(tf.nn.conv3d(inputs[i], kernels[i], strides=[1, stride[0], stride[1], stride[2], 1], padding=padding_tf, dilations=[1, dilation[0], dilation[1], dilation[2], 1]))

            result = tf.concat(convolved_parts, axis=4)

        if bias is not None:
            result = tf.add(result, bias)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    try:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    except Exception as e:
        print(f"Error: {e}")
        print(f"Torch result: {torch_result['result']}")
        print(f"Tensorflow result: {tf_result['result']}")
        raise

    input_data = {
        "input": np.random.rand(1, 6, 10, 10, 10).astype(np.float32),
        "weight": np.random.rand(5, 2, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    try:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    except Exception as e:
        print(f"Error: {e}")
        print(f"Torch result: {torch_result['result']}")
        print(f"Tensorflow result: {tf_result['result']}")
        raise
    
    print("Success")

if __name__ == "__main__":
    main()