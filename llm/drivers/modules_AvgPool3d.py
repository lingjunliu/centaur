import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict.get("kernel_size")
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    count_include_pad = input_dict.get("count_include_pad", True)
    divisor_override = input_dict.get("divisor_override", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    avg_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
    
    result = avg_pool(input_tensor)

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
        kernel_size = input_dict.get("kernel_size")
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        ceil_mode = input_dict.get("ceil_mode", False)
        count_include_pad = input_dict.get("count_include_pad", True)
        divisor_override = input_dict.get("divisor_override", None)
        
        if stride is None:
            stride = kernel_size

        if isinstance(padding, int):
            padding = [padding, padding, padding]
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size, kernel_size]

        if isinstance(stride, int):
             stride = [stride, stride, stride]
        
        tf_padding = "VALID"
        if padding != [0,0,0]:
            tf_padding = "SAME"
        
        
        if tf_padding == "SAME":
            paddings = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]], [0, 0]]
            input_tensor = tf.pad(input_tensor, paddings, "CONSTANT")
            tf_padding = "VALID"

        # Transpose the input tensor before applying the pooling operation
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1])

        result = tf.nn.avg_pool3d(
            input_tensor,
            ksize=[1, kernel_size[0], kernel_size[1], kernel_size[2], 1],
            strides=[1, stride[0], stride[1], stride[2], 1],
            padding=tf_padding
        )

        # Transpose the output tensor back to the original shape
        result = tf.transpose(result, perm=[0, 4, 1, 2, 3])
        

        result = result.numpy()

        if divisor_override is not None:
           result = result * (np.prod(kernel_size) / divisor_override)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()