import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict.get("output_size")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveAvgPool2d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    output_size = input_dict.get("output_size")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        channels = input_shape[3]

        target_height = output_size[0]
        target_width = output_size[1]

        input_height = tf.cast(input_shape[1], dtype=tf.float32)
        input_width = tf.cast(input_shape[2], dtype=tf.float32)

        target_height = tf.cast(target_height, dtype=tf.float32)
        target_width = tf.cast(target_width, dtype=tf.float32)
        
        stride_height = input_height / target_height
        stride_width = input_width / target_width
        
        kernel_height = input_height - (target_height - 1) * stride_height
        kernel_width = input_width - (target_width - 1) * stride_width

        kernel_height = tf.cast(kernel_height, dtype=tf.int32)
        kernel_width = tf.cast(kernel_width, dtype=tf.int32)

        stride_height = tf.cast(stride_height, dtype=tf.int32)
        stride_width = tf.cast(stride_width, dtype=tf.int32)

        result = tf.nn.avg_pool(
            input_tensor,
            ksize=[1, kernel_height, kernel_width, 1],
            strides=[1, stride_height, stride_width, 1],
            padding="VALID"
        )
        result = tf.image.resize(result, [output_size[0], output_size[1]])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 8, 8, 3).astype(np.float32),
        "output_size": (4, 4)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()