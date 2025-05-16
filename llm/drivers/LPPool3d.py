import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    norm_type = input_dict.get("norm_type", 2.0)
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    ceil_mode = input_dict.get("ceil_mode", False)

    if stride is None:
        stride = kernel_size

    if not cpu:
        input_tensor = input_tensor.cuda()

    pool = nn.LPPool3d(norm_type=norm_type, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)

    if not cpu:
        pool = pool.cuda()

    result = pool(input_tensor)

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
        ceil_mode = input_dict.get("ceil_mode", False)

        if stride is None:
            stride = kernel_size

        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        def lp_norm_pool3d(input_tensor, norm_type, kernel_size, stride, ceil_mode):

            ksize = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
            strides = [1, stride[0], stride[1], stride[2], 1]

            if norm_type == float('inf'):
                result = tf.nn.max_pool3d(input_tensor, ksize=ksize,
                                            strides=strides, padding='VALID')

            else:
                input_power = tf.pow(tf.abs(input_tensor), norm_type)
                pooled = tf.nn.avg_pool3d(input_power, ksize=ksize,
                                                strides=strides, padding='VALID')

                result = tf.pow(pooled, 1.0 / norm_type)


            result = tf.squeeze(result, axis=0)
            return result

        result = lp_norm_pool3d(input_tensor, norm_type, kernel_size, stride, ceil_mode)
        torch_result_shape = torch_version(input_dict)['result'].shape

        result = tf.image.resize(tf.expand_dims(result, axis=0), size=(torch_result_shape[1], torch_result_shape[2]), method='nearest')
        result = tf.squeeze(result, axis=0).numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 8, 8, 8).astype(np.float32),
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "norm_type": 2.0,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()