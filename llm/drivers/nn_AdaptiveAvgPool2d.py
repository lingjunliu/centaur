import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict.get("output_size")

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    m = nn.AdaptiveAvgPool2d(output_size)
    output = m(input_tensor)
    
    if not cpu:
        output = output.cpu()
    
    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        output_size = input_dict.get("output_size")

        if isinstance(output_size, int):
            target_height = output_size
            target_width = output_size
        elif isinstance(output_size, tuple):
            target_height = output_size[0] if output_size[0] is not None else input_tensor.shape[2]
            target_width = output_size[1] if output_size[1] is not None else input_tensor.shape[3]
        else:
            target_height = input_tensor.shape[2]
            target_width = input_tensor.shape[3]

        input_shape = tf.shape(input_tensor)
        batch_size, num_channels, height, width = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        def adaptive_avg_pool2d(input_tensor, output_size):
            target_height, target_width = output_size

            height_stride = tf.cast(height / target_height, dtype=tf.float32)
            width_stride = tf.cast(width / target_width, dtype=tf.float32)

            output = tf.TensorArray(dtype=tf.float32, size=target_height)

            for i in tf.range(target_height):
                row_output = tf.TensorArray(dtype=tf.float32, size=target_width)
                for j in tf.range(target_width):
                    h_start = tf.cast(tf.floor(tf.cast(i, dtype=tf.float32) * height_stride), dtype=tf.int32)
                    w_start = tf.cast(tf.floor(tf.cast(j, dtype=tf.float32) * width_stride), dtype=tf.int32)
                    h_end = tf.cast(tf.ceil(tf.cast(i + 1, dtype=tf.float32) * height_stride), dtype=tf.int32)
                    w_end = tf.cast(tf.ceil(tf.cast(j + 1, dtype=tf.float32) * width_stride), dtype=tf.int32)

                    h_end = tf.minimum(h_end, height)
                    w_end = tf.minimum(w_end, width)
                    
                    region = input_tensor[:, :, h_start:h_end, w_start:w_end]
                    avg_val = tf.reduce_mean(region, axis=[2, 3])
                    row_output = row_output.write(j, avg_val)
                output = output.write(i, row_output.stack())

            output = output.stack()
            output = tf.transpose(output, perm=[1, 2, 0])

            return output

        output_size_tuple = (target_height, target_width)
        output = adaptive_avg_pool2d(input_tensor, output_size_tuple)

        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(1, 64, 10, 9).astype(np.float32),
        "output_size": (5, 7)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.randn(1, 64, 10, 9).astype(np.float32),
        "output_size": 7
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.randn(1, 64, 10, 9).astype(np.float32),
        "output_size": (None, 7)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()