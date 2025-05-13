import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveMaxPool3d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        
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

            stride_depth = tf.cast(tf.math.floor(depth / output_depth), tf.int32)
            stride_height = tf.cast(tf.math.floor(height / output_height), tf.int32)
            stride_width = tf.cast(tf.math.floor(width / output_width), tf.int32)

            kernel_depth = depth - (output_depth - 1) * stride_depth
            kernel_height = height - (output_height - 1) * stride_height
            kernel_width = width - (output_width - 1) * stride_width
            
            
            pooled = tf.nn.max_pool3d(
                input=input_tensor,
                ksize=[1, kernel_depth, kernel_height, kernel_width, 1],
                strides=[1, stride_depth, stride_height, stride_width, 1],
                padding='VALID'
            )
            
            return pooled

        result = adaptive_max_pool3d(input_tensor, output_size)
        
        result = result.numpy()
    return {'result': result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 10, 10, 10).astype(np.float32),
        "output_size": (5, 7, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]
    
    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()