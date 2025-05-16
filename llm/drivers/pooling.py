import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", kernel_size)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.avg_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

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
        stride = input_dict.get("stride", kernel_size)
        padding = input_dict.get("padding", 0)

        ksize = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]

        if padding == 0:
            padding_type = 'VALID'
        else:
            padding_type = 'SAME'

        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])
        result = tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding=padding_type)
        result = tf.transpose(result, perm=[0, 3, 1, 2])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()