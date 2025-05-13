import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"])
    output_size = input_dict.get("output_size", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()

    result = torch.nn.MaxUnpool3d(kernel_size=input_dict["kernel_size"], stride=input_dict.get("stride", None), padding=input_dict.get("padding", 0))(input_tensor, indices, output_size=output_size)

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
        indices = tf.constant(input_dict["indices"])
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        output_size = input_dict.get("output_size", None)

        input_shape = tf.shape(input_tensor)
        batch_size = tf.cast(input_shape[0], tf.int64)
        channels = tf.cast(input_shape[1], tf.int64)
        input_depth = tf.cast(input_shape[2], tf.int64)
        input_height = tf.cast(input_shape[3], tf.int64)
        input_width = tf.cast(input_shape[4], tf.int64)

        if stride is None:
            stride = kernel_size
        
        if isinstance(padding, int):
            padding = [padding, padding, padding]

        if output_size is None:
            output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
            output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
            output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_size[2]
            output_size = [batch_size, channels, output_depth, output_height, output_width]
        else:
            output_size = [batch_size, channels, output_size[0], output_size[1], output_size[2]]

        updates = tf.reshape(input_tensor, [-1])
        
        indices_reshaped = tf.reshape(indices, [-1])

        depth_size = output_size[2]
        height_size = output_size[3]
        width_size = output_size[4]

        idx_base = tf.range(batch_size, dtype=tf.int64) * depth_size * height_size * width_size * channels
        idx_base = tf.reshape(idx_base, [tf.cast(input_shape[0], tf.int64), 1])

        idx = tf.cast(indices_reshaped, tf.int64) + idx_base

        output_shape = [tf.cast(input_shape[0], tf.int64) * channels * depth_size * height_size * width_size]
        sparse_tensor = tf.scatter_nd(indices=tf.expand_dims(idx, axis=1), updates=updates, shape=output_shape)
        sparse_tensor = tf.reshape(sparse_tensor, output_size)
        
        result = sparse_tensor.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 2, 3, 4, 5).astype(np.float32),
        "indices": np.random.randint(0, 3 * 4 * 5, size=(1, 2, 3, 4, 5)).astype(np.int64),
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "output_size": (5, 7, 9)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()