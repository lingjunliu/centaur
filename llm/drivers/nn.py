import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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

    result = torch.nn.functional.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

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
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]
        if isinstance(padding, int):
            padding = [padding, padding]

        strides = [1, stride[0], stride[1], 1]
        dilations = [1, dilation[0], dilation[1], 1]

        input_shape = input_tensor.shape
        weight_shape = weight.shape
        
        padding_tf = 'VALID' if padding == 0 else 'SAME'

        if groups > 1:
            input_channels = int(input_shape[3])
            weight_input_channels = int(weight_shape[1])
            
            if input_channels % groups != 0 :
                 raise ValueError("The number of input channels must be divisible by the number of groups.")
            if weight_input_channels * groups != input_channels:
                raise ValueError("Weight input channels * groups must equal input channels")

            weight_output_channels = int(weight_shape[0])

            input_tensor_split = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
            weight_split = tf.split(weight, num_or_size_splits=groups, axis=0)

            result_list = []
            for i in range(groups):
                result_list.append(tf.nn.conv2d(input_tensor_split[i], weight_split[i], strides=strides, padding=padding_tf, dilations=dilations))

            result = tf.concat(result_list, axis=3)

        else:
            
            if int(input_shape[3]) % int(weight_shape[1]) !=0:
                 print(f"Input depth {input_shape[3]} must be evenly divisible by filter depth {weight_shape[1]}")
                 return {"result": np.zeros_like(input_dict['input'])}
            result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_tf, dilations=dilations)

        if bias is not None:
            result = tf.nn.bias_add(result, bias)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
    }
    try:
        torch_result = torch_version(input_data)
        tf_result = tensorflow_version(input_data)

        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

        print("Success")
    except ValueError as e:
         print(e)

if __name__ == "__main__":
    main()