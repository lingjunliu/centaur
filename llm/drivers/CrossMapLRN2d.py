import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", 5)
    alpha = input_dict.get("alpha", 0.0001)
    beta = input_dict.get("beta", 0.75)
    k = input_dict.get("k", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    lrn = nn.CrossMapLRN2d(size=size, alpha=alpha, beta=beta, k=k)
    result = lrn(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        size = input_dict.get("size", 5)
        alpha = input_dict.get("alpha", 0.0001)
        beta = input_dict.get("beta", 0.75)
        k = input_dict.get("k", 1.0)

        input_shape = input_tensor.shape
        depth = input_shape[-1]
        side = size // 2

        padded_tensor = tf.pad(input_tensor, [[0, 0], [0, 0], [0, 0], [side, side]], mode="CONSTANT")

        def lrn_channel(i):
            start = max(0, i - side)
            end = min(depth, i + side + 1)
            
            slice_begin = [0, 0, 0, start]
            slice_size = [-1, -1, -1, end - start]
            
            channel_slice = tf.slice(padded_tensor, slice_begin, slice_size)
            squared_slice = tf.square(channel_slice)
            sum_channels = tf.reduce_sum(squared_slice, axis=3, keepdims=True)
            return sum_channels

        lrn_list = []
        for i in range(depth):
          lrn_list.append(lrn_channel(i))

        lrn = tf.concat(lrn_list, axis=3)
        result = input_tensor / tf.pow(k + alpha * lrn, beta)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "size": 5,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()