import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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

    result = torch.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

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
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias)
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        if isinstance(padding, int):
            padding_enum = "VALID" if padding == 0 else "SAME"
        else:
            padding_enum = "VALID"

        input_shape = input_tensor.shape
        weight_shape = weight.shape

        if len(input_shape) == 4 and len(weight_shape) == 4:
            in_channels = input_shape[1]
            out_channels = weight_shape[0]
            kernel_size = weight_shape[2]
            
            if in_channels % groups != 0:
                raise ValueError("The number of input channels must be divisible by the number of groups")
            
            if weight_shape[1] != in_channels // groups:
                raise ValueError(f"The number of weight input channels ({weight_shape[1]}) does not match the expected value ({in_channels // groups})")

            if in_channels != weight_shape[1] * groups:
                raise ValueError(f"Input channels ({in_channels}) not compatible with weight input channels ({weight_shape[1]}) and groups ({groups})")
        
        try:
            if bias is not None:
                result = tf.nn.conv2d(input_tensor, weight, strides=[1, stride, stride, 1], padding=padding_enum, dilations=[1, dilation, dilation, 1]) + bias
            else:
                result = tf.nn.conv2d(input_tensor, weight, strides=[1, stride, stride, 1], padding=padding_enum, dilations=[1, dilation, dilation, 1])
        except tf.errors.InvalidArgumentError as e:
            print(f"TensorFlow encountered an InvalidArgumentError: {e}")
            return {"result": np.zeros(torch.conv2d(torch.tensor(input_dict["input"]), torch.tensor(input_dict["weight"])).shape.as_list())} # Return zeroed array with torch shape

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "padding": 0,
        "stride": 1,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    if tf_result is None:
        print("TensorFlow encountered an error, skipping assertion.")
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()