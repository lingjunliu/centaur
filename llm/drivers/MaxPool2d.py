import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", kernel_size)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    return_indices = input_dict.get("return_indices", False)
    ceil_mode = input_dict.get("ceil_mode", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    
    result = max_pool(input_tensor)
    
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
        dilation = input_dict.get("dilation", 1)
        ceil_mode = input_dict.get("ceil_mode", False)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        ksize = [1, kernel_size[0], kernel_size[1], 1]
        strides = [1, stride[0], stride[1], 1]

        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'

        if ceil_mode:
            input_shape = input_tensor.shape
            output_height = np.ceil((input_shape[1] - ksize[1] + 1 + 2 * padding[0]) / strides[1])
            output_width = np.ceil((input_shape[2] - ksize[2] + 1 + 2 * padding[1]) / strides[2])
            pad_along_height = max(0, (output_height - 1) * strides[0] + ksize[0] - input_shape[1] - 2 * padding[0])
            pad_along_width = max(0, (output_width - 1) * strides[1] + ksize[1] - input_shape[2] - 2 * padding[1])
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
            input_tensor = tf.pad(input_tensor, paddings, "CONSTANT")

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])
        result = tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding=padding_tf, data_format='NHWC')
        result = tf.transpose(result, perm=[0, 3, 1, 2])


        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()