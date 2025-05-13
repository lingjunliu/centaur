import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    x = torch.tensor(input_dict["x"])
    y = torch.tensor(input_dict["y"])
    dim = input_dict.get("dim", -1)

    if not cpu:
        x = x.cuda()
        y = y.cuda()

    result = torch.linalg.vecdot(x, y, dim=dim)

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
        x = tf.constant(input_dict["x"])
        y = tf.constant(input_dict["y"])
        dim = input_dict.get("dim", -1)

        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        rank = tf.rank(x)

        dim = dim % rank

        x_reshaped = tf.transpose(x, perm=tf.concat([tf.range(dim), [rank - 1], tf.range(dim, rank - 1)], axis=0))
        y_reshaped = tf.transpose(y, perm=tf.concat([tf.range(dim), [rank - 1], tf.range(dim, rank - 1)], axis=0))
        
        x_shape_reshaped = tf.shape(x_reshaped)
        y_shape_reshaped = tf.shape(y_reshaped)

        x_reshaped = tf.reshape(x_reshaped, tf.concat([tf.reduce_prod(x_shape_reshaped[:rank-1], keepdims=True), [x_shape_reshaped[rank-1]]], axis=0))
        y_reshaped = tf.reshape(y_reshaped, tf.concat([tf.reduce_prod(y_shape_reshaped[:rank-1], keepdims=True), [y_shape_reshaped[rank-1]]], axis=0))

        result = tf.reduce_sum(tf.math.conj(x_reshaped) * y_reshaped, axis=1)

        new_shape = tf.concat([x_shape[:dim], x_shape[dim:-1]], axis=0)
        result = tf.reshape(result, new_shape)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "x": np.random.rand(3, 4, 5).astype(np.float32),
        "y": np.random.rand(3, 4, 5).astype(np.float32),
        "dim": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()