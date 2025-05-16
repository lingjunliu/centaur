import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", 5)
    alpha = input_dict.get("alpha", 0.0001)
    beta = input_dict.get("beta", 0.75)
    k = input_dict.get("k", 1.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    lrn = nn.CrossMapLRN2d(size, alpha, beta, k)
    
    if not cpu:
        lrn = lrn.cuda()
        
    input_tensor = input_tensor.reshape(1, 1, int(np.sqrt(input_tensor.shape[0])), int(np.sqrt(input_tensor.shape[0])))
    result = lrn(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        size = input_dict.get("size", 5)
        alpha = input_dict.get("alpha", 0.0001)
        beta = input_dict.get("beta", 0.75)
        k = input_dict.get("k", 1.0)
        
        input_size = input_tensor.shape[0]
        side_len = int(np.sqrt(input_size))
        input_tensor = tf.reshape(input_tensor, [1, side_len, side_len, 1])
        
        def cross_map_lrn(input_tensor, size, alpha, beta, k):
            half_size = size // 2
            input_sqr = tf.square(input_tensor)
            
            padding = [[0, 0], [half_size, half_size], [half_size, half_size], [0, 0]]
            input_sqr_padded = tf.pad(input_sqr, padding, "CONSTANT")
            
            pooled_sqr = tf.nn.avg_pool(input_sqr_padded, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='VALID')
            
            lrn = input_tensor / tf.pow(k + alpha * pooled_sqr, beta)
            return lrn
        
        result = cross_map_lrn(input_tensor, size, alpha, beta, k)
        
        result = result.numpy().squeeze()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_size = 16
    input_data = {
        "input": np.random.rand(input_size).astype(np.float32),
        "size": 5,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()