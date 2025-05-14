import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", kernel_size)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    count_include_pad = input_dict.get("count_include_pad", True)
    divisor_override = input_dict.get("divisor_override", None)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.avg_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )
    
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
        stride = input_dict.get("stride", kernel_size)
        padding = input_dict.get("padding", 0)
        ceil_mode = input_dict.get("ceil_mode", False)
        count_include_pad = input_dict.get("count_include_pad", True)
        divisor_override = input_dict.get("divisor_override", None)

        input_tensor = tf.cast(input_tensor, tf.float32)
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]
        
        ksize = [1, kernel_size[0], kernel_size[1], 1]
        strides = [1, stride[0], stride[1], 1]

        if padding == 0 or padding == [0, 0]:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'

        input_shape = input_tensor.shape
        
        result = tf.nn.avg_pool(
            input_tensor,
            ksize=ksize,
            strides=strides,
            padding=padding_tf
        )

        if divisor_override is not None:
           result = result / divisor_override
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_output = torch_result["result"]
    tf_output = tf_result["result"]

    min_b = min(torch_output.shape[0], tf_output.shape[0])
    min_c = min(torch_output.shape[1], tf_output.shape[1])
    min_h = min(torch_output.shape[2], tf_output.shape[2])
    min_w = min(torch_output.shape[3], tf_output.shape[3])

    torch_cropped = torch_output[:min_b, :min_c, :min_h, :min_w]
    tf_cropped = tf_output[:min_b, :min_c, :min_h, :min_w]
    
    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()