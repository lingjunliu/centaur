import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    norm_type = input_dict.get("norm_type", 2)
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    if stride is None:
        stride = kernel_size
    ceil_mode = input_dict.get("ceil_mode", False)

    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.lp_pool1d(input_tensor, norm_type, kernel_size, stride, ceil_mode=ceil_mode)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    norm_type = input_dict.get("norm_type", 2)
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    if stride is None:
        stride = kernel_size
    ceil_mode = input_dict.get("ceil_mode", False)

    if len(input_tensor.shape) == 1:
        input_tensor = tf.reshape(input_tensor, (1, 1, -1, 1))
    elif len(input_tensor.shape) == 2:
        input_tensor = tf.reshape(input_tensor, (1, input_tensor.shape[0], input_tensor.shape[1], 1))
    elif len(input_tensor.shape) == 3:
        input_tensor = tf.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], 1))
    elif len(input_tensor.shape) == 4:
        pass
    else:
        raise ValueError("Input tensor must have 1, 2, 3 or 4 dimensions")

    padding = 'VALID'
    if ceil_mode:
      padding = 'SAME'

    def lp_norm(x, p):
        return tf.pow(tf.reduce_sum(tf.pow(tf.abs(x), p), axis=1, keepdims=False), 1.0 / p)

    def pool_1d(input_tensor, kernel_size, stride, padding, norm_type):
        patches = tf.image.extract_patches(
            images=input_tensor,
            sizes=[1, 1, kernel_size, 1],
            strides=[1, 1, stride, 1],
            rates=[1, 1, 1, 1],
            padding=padding
        )

        patches = tf.reshape(patches, [-1, kernel_size])

        lp_norms = lp_norm(patches, norm_type)

        if ceil_mode:
          output_length = tf.cast(tf.math.ceil(tf.cast(tf.shape(input_tensor)[2], dtype=tf.float32) / stride), dtype=tf.int32)
          current_length = tf.shape(lp_norms)[0]
          padding_size = tf.maximum(0, output_length - current_length)
          lp_norms = tf.pad(lp_norms, [[0, padding_size]])

        return lp_norms

    result = pool_1d(input_tensor, kernel_size, stride, padding, norm_type)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
        "kernel_size": 3,
        "stride": 2,
        "norm_type": 2,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
        "kernel_size": 3,
        "stride": 2,
        "norm_type": 2,
        "ceil_mode": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
        "kernel_size": 3,
        "norm_type": 2,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
        "kernel_size": 3,
        "norm_type": 2,
        "ceil_mode": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")