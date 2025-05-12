import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.nn.functional.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)
    result = torch.nn.functional.relu(result)

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
        bias = tf.constant(input_dict["bias"])
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        
        if isinstance(padding, int):
            padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
        else:
            padding = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]

        if isinstance(stride, int):
            strides = [1, stride, stride, 1]
        else:
            strides = [1, stride[0], stride[1], 1]
            
        if isinstance(dilation, int):
            rates = [1, dilation, dilation, 1]
        else:
            rates = [1, dilation[0], dilation[1], 1]
            
        input_shape = input_tensor.shape.as_list()
        weight_shape = weight.shape.as_list()
        
        if input_shape[-1] % groups != 0:
            raise ValueError("Input depth must be evenly divisible by groups.")
        if weight_shape[1] * groups != input_shape[1]:
            raise ValueError(f"Weight input channels {weight_shape[1] * groups} must match the input depth {input_shape[1]} when groups is {groups}.")

        if groups == 1:
            conv = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding, dilations=rates)
        else:
            input_splits = []
            weight_splits = []
            try:
                input_splits = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
                weight_splits = tf.split(weight, num_or_size_splits=groups, axis=0)
            except Exception as e:
                print(f"Error during split: {e}")
                raise

            conv_outputs = []
            for i in range(groups):
                conv_outputs.append(tf.nn.conv2d(input_splits[i], weight_splits[i], strides=strides, padding=padding, dilations=rates))
            
            conv = tf.concat(conv_outputs, axis=3)
            
        result = tf.nn.relu(tf.nn.bias_add(conv, bias))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(6, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "padding": 2,
        "stride": 1,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    
    #Modify the weight to be valid for tensorflow
    input_channels = input_data["input"].shape[1]
    input_data["weight"] = np.random.rand(6, input_channels, 5, 5).astype(np.float32)
    
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()