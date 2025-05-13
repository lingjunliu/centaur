import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d(output_size)
    result = adaptive_avg_pool3d(input_tensor)
    
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
        output_size = input_dict["output_size"]
        
        target_d, target_h, target_w = output_size
        
        input_shape = tf.shape(input_tensor)
        input_d = tf.cast(input_shape[1], dtype=tf.float32)
        input_h = tf.cast(input_shape[2], dtype=tf.float32)
        input_w = tf.cast(input_shape[3], dtype=tf.float32)
        input_c = tf.cast(input_shape[4], dtype=tf.float32)

        stride_d = input_d / tf.cast(target_d, dtype=tf.float32)
        stride_h = input_h / tf.cast(target_h, dtype=tf.float32)
        stride_w = input_w / tf.cast(target_w, dtype=tf.float32)

        kernel_d = input_d / tf.cast(target_d, dtype=tf.float32)
        kernel_h = input_h / tf.cast(target_h, dtype=tf.float32)
        kernel_w = input_w / tf.cast(target_w, dtype=tf.float32)

        stride_d = tf.cast(tf.math.floor(stride_d), dtype=tf.int32)
        stride_h = tf.cast(tf.math.floor(stride_h), dtype=tf.int32)
        stride_w = tf.cast(tf.math.floor(stride_w), dtype=tf.int32)

        kernel_d = tf.cast(tf.math.ceil(kernel_d), dtype=tf.int32)
        kernel_h = tf.cast(tf.math.ceil(kernel_h), dtype=tf.int32)
        kernel_w = tf.cast(tf.math.ceil(kernel_w), dtype=tf.int32)

        result = tf.nn.avg_pool3d(
            input_tensor,
            ksize=[1, kernel_d, kernel_h, kernel_w, 1],
            strides=[1, stride_d, stride_h, stride_w, 1],
            padding='VALID'
        )
        
        result = tf.transpose(result, perm=[0, 1, 4, 2, 3])
        s = tf.shape(result)
        result = tf.reshape(result, [s[0], s[1], s[2], s[3] * s[4]])
        result = tf.image.resize(result, [target_d, target_h])
        result = tf.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 8, 16, 32, 64).astype(np.float32),
        "output_size": (4, 8, 16)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()