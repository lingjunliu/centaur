import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveAvgPool1d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        input_length = tf.cast(input_shape[-1], tf.float32)
        
        def adaptive_avg_pool1d(input_tensor, output_size):
            input_shape = tf.shape(input_tensor)
            input_length = tf.cast(input_shape[-1], tf.float32)
            output_size = output_size if isinstance(output_size, int) else output_size[0]
            
            stride = tf.cast(tf.math.floor(input_length / output_size), tf.int32)
            kernel_size = tf.cast(input_length - (output_size - 1) * tf.cast(stride, tf.float32), tf.int32)
            
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=1)
            
            result = tf.nn.avg_pool(
                input_tensor,
                ksize=[1, 1, kernel_size, 1],
                strides=[1, 1, stride, 1],
                padding='VALID'
            )
            
            result = tf.squeeze(result, axis=[0, 1])
            return result
        
        result = adaptive_avg_pool1d(input_tensor, output_size)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32),
        "output_size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=np.float32),
        "output_size": 2
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]], dtype=np.float32),
        "output_size": 4
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()