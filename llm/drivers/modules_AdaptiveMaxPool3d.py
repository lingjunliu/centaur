import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    adaptive_max_pool3d = nn.AdaptiveMaxPool3d(output_size)

    if not cpu:
        adaptive_max_pool3d = adaptive_max_pool3d.cuda()

    result = adaptive_max_pool3d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    input_shape = input_tensor.shape
    input_tensor = tf.cast(input_tensor, dtype=tf.float32)

    if len(input_shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_shape = input_tensor.shape
    if len(input_shape) == 4:
        input_tensor = tf.expand_dims(input_tensor, axis=1)

    def adaptive_max_pool3d(input_tensor, output_size):
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        channels = input_shape[1]
        depth = input_shape[2]
        height = input_shape[3]
        width = input_shape[4]

        output_depth = output_size[0]
        output_height = output_size[1]
        output_width = output_size[2]

        def calculate_indices(input_size, output_size):
            ratio = tf.cast(input_size, tf.float32) / tf.cast(output_size, tf.float32)
            indices = tf.cast(tf.cast(tf.range(output_size), tf.float32) * ratio, tf.int32)
            return indices

        depth_indices = calculate_indices(depth, output_depth)
        height_indices = calculate_indices(height, output_height)
        width_indices = calculate_indices(width, output_width)

        output_tensor = tf.zeros((batch_size, channels, output_depth, output_height, output_width), dtype=tf.float32)
        
        for b in range(batch_size):
            for c in range(channels):
                for od in range(output_depth):
                    for oh in range(output_height):
                        for ow in range(output_width):
                            d_start = depth_indices[od]
                            d_end = depth_indices[od] + tf.cast(tf.math.ceil(tf.cast(depth, tf.float32) / tf.cast(output_depth, tf.float32)), tf.int32)
                            d_end = tf.minimum(d_end, depth)
                            
                            h_start = height_indices[oh]
                            h_end = height_indices[oh] + tf.cast(tf.math.ceil(tf.cast(height, tf.float32) / tf.cast(output_height, tf.float32)), tf.int32)
                            h_end = tf.minimum(h_end, height)

                            w_start = width_indices[ow]
                            w_end = width_indices[ow] + tf.cast(tf.math.ceil(tf.cast(width, tf.float32) / tf.cast(output_width, tf.float32)), tf.int32)
                            w_end = tf.minimum(w_end, width)

                            slice_tensor = input_tensor[b, c, d_start:d_end, h_start:h_end, w_start:w_end]
                            max_val = tf.reduce_max(slice_tensor)
                            output_tensor = tf.tensor_scatter_nd_update(output_tensor, [[b, c, od, oh, ow]], [max_val])

        return output_tensor

    result = adaptive_max_pool3d(input_tensor, output_size)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 10, 12, 14).astype(np.float32),
        "output_size": (5, 6, 7)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()