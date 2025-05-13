import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    count_include_pad = input_dict.get("count_include_pad", True)
    divisor_override = input_dict.get("divisor_override", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    avg_pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, 
                                    ceil_mode=ceil_mode, count_include_pad=count_include_pad, 
                                    divisor_override=divisor_override)
    
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
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        ceil_mode = input_dict.get("ceil_mode", False)
        count_include_pad = input_dict.get("count_include_pad", True)
        divisor_override = input_dict.get("divisor_override", None)

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        
        if isinstance(stride, int):
            stride = [stride, stride]
        
        if isinstance(padding, int):
            padding = [padding, padding]

        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        
        input_shape = input_tensor.shape
        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        if len(input_shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        pad_list = [[0,0], [0,0], [0,0], [0,0]]
        pad_list[1][0] = padding[0]
        pad_list[1][1] = padding[0]
        pad_list[2][0] = padding[1]
        pad_list[2][1] = padding[1]
            
        if padding != [0, 0]:
            input_tensor = tf.pad(input_tensor, pad_list, "CONSTANT")

        if divisor_override is not None:
            result = tf.nn.avg_pool(
                input_tensor,
                ksize=[1, kernel_size[0], kernel_size[1], 1],
                strides=[1, stride[0], stride[1], 1],
                padding='VALID'
            )
            result = result / divisor_override
        else:
            result = tf.nn.avg_pool(
                input_tensor,
                ksize=[1, kernel_size[0], kernel_size[1], 1],
                strides=[1, stride[0], stride[1], 1],
                padding='VALID'
            )

        if not count_include_pad:
            result = tf.where(tf.math.is_nan(result), tf.zeros_like(result), result)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32),
        "kernel_size": 2,
        "stride": 1,
        "padding": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32),
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32),
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32),
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (1, 1),
        "count_include_pad": False
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()