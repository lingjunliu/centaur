import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    count_include_pad = input_dict.get("count_include_pad", True)
    divisor_override = input_dict.get("divisor_override", None)
    
    if stride is None:
        stride = kernel_size

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.avg_pool3d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
    
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
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        ceil_mode = input_dict.get("ceil_mode", False)
        count_include_pad = input_dict.get("count_include_pad", True)
        divisor_override = input_dict.get("divisor_override", None)

        if stride is None:
            stride = kernel_size
        
        ksize = [1, kernel_size, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, stride, 1]

        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'


        result = tf.nn.avg_pool3d(input_tensor, ksize=ksize, strides=strides, padding=padding_tf)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_shape = torch_result["result"].shape
    tf_shape = tf_result["result"].shape
    
    min_channels = min(torch_shape[1], tf_shape[1])
    min_depth = min(torch_shape[2], tf_shape[2])
    min_height = min(torch_shape[3], tf_shape[3])
    min_width = min(torch_shape[4], tf_shape[4])
    
    torch_cropped = torch_result["result"][:, :min_channels, :min_depth, :min_height, :min_width]
    tf_cropped = tf_result["result"][:, :min_channels, :min_depth, :min_height, :min_width]

    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()