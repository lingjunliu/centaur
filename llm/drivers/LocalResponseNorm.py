import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", 5)
    alpha = input_dict.get("alpha", 0.0001)
    beta = input_dict.get("beta", 0.75)
    k = input_dict.get("k", 1.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    lrm = torch.nn.LocalResponseNorm(size, alpha, beta, k)
    result = lrm(input_tensor)
    
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
        size = input_dict.get("size", 5)
        alpha = input_dict.get("alpha", 0.0001)
        beta = input_dict.get("beta", 0.75)
        k = input_dict.get("k", 1.0)
        
        input_shape = input_tensor.shape
        if len(input_shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        elif len(input_shape) == 1:
            input_tensor = tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=0)

        squared = tf.square(input_tensor)
        half_size = size // 2

        def loop_body(i, accumulated_sum):
            slice_start = [0, i, 0]
            slice_size = [-1, 1, -1]
            current_slice = squared[:, i:i+1, :]
            accumulated_sum = tf.add(accumulated_sum, current_slice)
            return i + 1, accumulated_sum

        i = tf.constant(0)
        accumulated_sum = tf.zeros_like(squared)

        _, accumulated_sum = tf.while_loop(
            cond=lambda i, _: i < tf.shape(squared)[1],
            body=loop_body,
            loop_vars=(i, accumulated_sum)
        )
        
        padding_before = half_size
        padding_after = size - 1 - half_size
        paddings = [[0, 0], [padding_before, padding_after], [0, 0]]
        padded_squared = tf.pad(squared, paddings, "CONSTANT")
        
        def loop_body2(i, summed_slice):
            begin = [0, i, 0]
            size2 = [-1, size, -1]
            current_slice = tf.slice(padded_squared, begin, size2)
            summed = tf.reduce_sum(current_slice, axis=1, keepdims=True)
            indices = [[0, i]]
            updates = summed[0, 0, :]
            
            summed_slice = tf.tensor_scatter_nd_update(summed_slice, indices, updates[tf.newaxis, :])
            return i + 1, summed_slice

        i = tf.constant(0)
        summed_slice = tf.zeros_like(squared)

        _, summed_slice = tf.while_loop(
            cond=lambda i, _: i < tf.shape(squared)[1],
            body=loop_body2,
            loop_vars=(i, summed_slice)
        )

        scale = tf.pow(k + alpha * summed_slice, -beta)
        result = input_tensor * scale

        if len(input_shape) == 2:
            result = tf.squeeze(result, axis=0)
        elif len(input_shape) == 1:
            result = tf.squeeze(tf.squeeze(result, axis=0), axis=0)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[-0.2144, -0.0841,  0.0367],
                            [ 0.4062, -0.4883, -0.6791]],

                           [[ 0.8160, -0.5561, -0.3627],
                            [ 1.1844, -0.0796, -0.9384]],

                           [[ 0.2712,  0.3597,  0.3297],
                            [-0.5909,  0.0823, -0.8180]]], dtype=np.float32),
        "size": 5,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()