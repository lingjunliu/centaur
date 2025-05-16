import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    norm_type = input_dict.get("norm_type", 2.0)
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    if stride is not None:
        stride = input_dict["stride"]
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.lp_pool2d(input_tensor, norm_type, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

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
        norm_type = input_dict.get("norm_type", 2.0)
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        ceil_mode = input_dict.get("ceil_mode", False)

        if stride is None:
            stride = kernel_size
            
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        input_shape = input_tensor.shape
        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])

        def lp_norm(x, norm_type):
            return tf.pow(tf.reduce_sum(tf.pow(tf.abs(x), norm_type), axis=(1, 2), keepdims=True), 1.0 / norm_type)
        
        ksize = [1, kernel_size[0], kernel_size[1], 1]
        strides = [1, stride[0], stride[1], 1]
        
        if padding != (0, 0):
            input_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], "CONSTANT")

        result = tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding='VALID')
        result = tf.pow(tf.reduce_sum(tf.pow(tf.abs(result), norm_type), axis=(1, 2), keepdims=True), 1.0 / norm_type)
        result = tf.transpose(result, perm=[0, 3, 1, 2])

        result = result.numpy()

        return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 32, 32).astype(np.float32),
        "norm_type": 2.0,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()