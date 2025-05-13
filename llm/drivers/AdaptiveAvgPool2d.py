import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size)
    result = adaptive_avg_pool2d(input_tensor)

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
        output_size = input_dict["output_size"]

        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        channels = input_shape[1]
        height = tf.cast(input_shape[2], tf.float32)
        width = tf.cast(input_shape[3], tf.float32)

        target_height = output_size[0]
        target_width = output_size[1]

        stride_height = tf.cast(tf.math.floor(height / target_height), tf.int32)
        stride_width = tf.cast(tf.math.floor(width / target_width), tf.int32)
        kernel_height = tf.cast(height - tf.cast((target_height - 1), tf.float32) * tf.cast(stride_height, tf.float32), tf.int32)
        kernel_width = tf.cast(width - tf.cast((target_width - 1), tf.float32) * tf.cast(stride_width, tf.float32), tf.int32)
        
        result = tf.nn.avg_pool(
            input_tensor,
            ksize=[1, kernel_height, kernel_width, 1],
            strides=[1, stride_height, stride_width, 1],
            padding='VALID'
        )
        
        result = tf.image.resize(result, (target_height, target_width))
        
        
        result = tf.transpose(result, perm=[0, 3, 1, 2])

        
        result = tf.transpose(result, perm=[0, 2, 3, 1])

        
        result = tf.transpose(result, perm=[0, 3, 1, 2])
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "output_size": (16, 16)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()