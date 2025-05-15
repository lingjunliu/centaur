import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = torch.tensor(input_dict["weight"])
    bias_tensor = torch.tensor(input_dict["bias"])
    
    padding = input_dict.get("padding", (0,))
    stride = input_dict.get("stride", (1,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)
    alpha = input_dict.get("alpha", 1.0)
    relu = input_dict.get("relu", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        bias_tensor = bias_tensor.cuda()

    conv_result = torch.nn.functional.conv2d(
        input_tensor.unsqueeze(0),
        weight_tensor,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).squeeze(0)

    result = input_tensor + alpha * conv_result
    
    if relu:
        result = torch.relu(result)

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
        weight_tensor = tf.constant(input_dict["weight"])
        bias_tensor = tf.constant(input_dict["bias"])

        padding = input_dict.get("padding", (0,))
        stride = input_dict.get("stride", (1,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)
        alpha = input_dict.get("alpha", 1.0)
        relu = input_dict.get("relu", False)

        if isinstance(padding, int):
            padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
        else:
            padding = [[0, 0], [padding[0], padding[0]], [padding[0], padding[0]], [0, 0]]
        
        input_tensor_expanded = tf.expand_dims(input_tensor, 0)
        
        if groups == 1:
            conv_result = tf.nn.conv2d(
                input_tensor_expanded,
                weight_tensor,
                strides=[1, stride[0], stride[0], 1],
                padding='VALID',
                dilations=[1, dilation[0], dilation[0], 1]
            )
            
            if padding != [[0, 0], [0, 0], [0, 0], [0, 0]]:
               pad_before = padding[1][0]
               pad_after = padding[1][1]
               conv_result = tf.pad(conv_result, [[0,0], [pad_before, pad_after], [pad_before, pad_after], [0,0]])
            conv_result = tf.nn.bias_add(conv_result, bias_tensor)
            
            added_result = input_tensor_expanded + tf.multiply(conv_result, alpha)

        else:
            input_splits = tf.split(input_tensor_expanded, groups, axis=3)
            weight_splits = tf.split(weight_tensor, groups, axis=0)
            
            conv_results = []
            for i in range(groups):
                conv = tf.nn.conv2d(
                    input_splits[i],
                    weight_splits[i],
                    strides=[1, stride[0], stride[0], 1],
                    padding='VALID',
                    dilations=[1, dilation[0], dilation[0], 1]
                )
                if padding != [[0, 0], [0, 0], [0, 0], [0, 0]]:
                   pad_before = padding[1][0]
                   pad_after = padding[1][1]
                   conv = tf.pad(conv, [[0,0], [pad_before, pad_after], [pad_before, pad_after], [0,0]])
                bias_split = bias_tensor[i * (bias_tensor.shape[0] // groups) : (i+1) * (bias_tensor.shape[0] // groups)]
                conv = tf.nn.bias_add(conv, bias_split)
                conv_results.append(conv)
            conv_result = tf.concat(conv_results, axis=3)
            added_result = input_tensor_expanded + tf.multiply(conv_result, alpha)
        
        if relu:
            added_result = tf.nn.relu(added_result)
        
        result = added_result.numpy()[0]

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "padding": (2, 2),
        "stride": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "alpha": 1.0,
        "relu": True,
    }
    
    input_data["input"] = np.random.rand(3, 32, 32).astype(np.float32)
    input_data["weight"] = np.random.rand(16, 3, 5, 5).astype(np.float32)
    input_data["bias"] = np.random.rand(16).astype(np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()