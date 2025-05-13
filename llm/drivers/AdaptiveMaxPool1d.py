import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    adaptive_max_pool1d = torch.nn.AdaptiveMaxPool1d(output_size)
    if not cpu:
        adaptive_max_pool1d = adaptive_max_pool1d.cuda()
    
    result = adaptive_max_pool1d(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    output_size = input_dict["output_size"]
    
    input_shape = tf.shape(input_tensor)
    
    if len(input_shape) == 1:
        input_tensor = tf.reshape(input_tensor, (1, input_shape[0]))

    def adaptive_max_pool1d_tf(input_tensor, output_size):
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        input_length = input_shape[1]

        stride = tf.cast(input_length / output_size, tf.float32)
        ksize = tf.cast(tf.cast(input_length, tf.float32) - (tf.cast(output_size, tf.float32) - 1) * tf.floor(stride), tf.int32)
        stride = tf.cast(tf.floor(stride), tf.int32)
        
        ksize = tf.maximum(1, ksize)
        stride = tf.maximum(1, stride)

        pooled = tf.nn.pool(input_tensor[..., None], window_shape=[ksize], pooling_type='MAX', strides=[stride], padding='VALID')

        pooled = tf.reshape(pooled, (batch_size, output_size))
        return pooled

    result = adaptive_max_pool1d_tf(input_tensor, output_size)

    if len(input_shape) == 1:
        result = tf.reshape(result, (output_size,))
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "output_size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()