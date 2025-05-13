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
        input_tensor = tf.constant(input_dict["input"])
        output_size = input_dict["output_size"]
        
        input_shape = tf.shape(input_tensor)
        height = input_shape[1]
        width = input_shape[2]
        
        target_height, target_width = output_size

        stride_height = max(1, height // target_height)
        stride_width = max(1, width // target_width)
        
        kernel_height = height - (target_height - 1) * stride_height
        kernel_width = width - (target_width - 1) * stride_width
        
        result = tf.nn.avg_pool(
            input_tensor[tf.newaxis, ...],
            ksize=[1, max(1, kernel_height), max(1, kernel_width), 1],
            strides=[1, stride_height, stride_width, 1],
            padding='VALID'
        )
        
        result = tf.reshape(result, [target_height, target_width])

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]], dtype=np.float32),
        "output_size": (2, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]

    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()