import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    correction = input_dict.get("correction", 1)
    fweights = input_dict.get("fweights", None)
    aweights = input_dict.get("aweights", None)

    if fweights is not None:
        fweights = torch.tensor(fweights)
    if aweights is not None:
        aweights = torch.tensor(aweights)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if fweights is not None:
            fweights = fweights.cuda()
        if aweights is not None:
            aweights = aweights.cuda()
    
    result = torch.cov(input_tensor, correction=correction, fweights=fweights, aweights=aweights)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    correction = input_dict.get("correction", 1)
    fweights = input_dict.get("fweights", None)
    aweights = input_dict.get("aweights", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        
        if len(input_tensor.shape) == 0:
            return {"result": tf.constant(0.0).numpy()}
        
        if len(input_tensor.shape) == 1:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        num_vars = input_tensor.shape[0]
        num_obs = input_tensor.shape[1]

        if fweights is not None and aweights is not None:
            fweights = tf.cast(fweights, dtype=tf.float32)
            aweights = tf.cast(aweights, dtype=tf.float32)
            weights = fweights * aweights
            weights_sum = tf.reduce_sum(weights)
            weighted_means = tf.reduce_sum(input_tensor * weights, axis=1, keepdims=True) / weights_sum
            centered_data = input_tensor - weighted_means
            cov_matrix = tf.matmul(centered_data, centered_data, transpose_b=True)
            w_a = tf.reduce_sum((weights*weights))
            delta_n = w_a / weights_sum
            denom = tf.maximum(0.0, weights_sum - delta_n * tf.cast(correction,dtype=tf.float32))

            cov_matrix = cov_matrix / denom
        elif fweights is not None:
            fweights = tf.cast(fweights, dtype=tf.float32)
            weights = fweights
            weights_sum = tf.reduce_sum(weights)
            weighted_means = tf.reduce_sum(input_tensor * weights, axis=1, keepdims=True) / weights_sum
            centered_data = input_tensor - weighted_means
            cov_matrix = tf.matmul(centered_data, centered_data, transpose_b=True)
            denom = tf.maximum(0.0, weights_sum - tf.cast(correction,dtype=tf.float32))
            cov_matrix = cov_matrix / denom
        elif aweights is not None:
            aweights = tf.cast(aweights, dtype=tf.float32)
            weights = aweights
            weights_sum = tf.reduce_sum(weights)
            weighted_means = tf.reduce_sum(input_tensor * weights, axis=1, keepdims=True) / weights_sum
            centered_data = input_tensor - weighted_means
            cov_matrix = tf.matmul(centered_data, centered_data, transpose_b=True)
            denom = tf.maximum(0.0, weights_sum - tf.cast(correction,dtype=tf.float32))
            cov_matrix = cov_matrix / denom
        else:
            means = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
            centered_data = input_tensor - means
            cov_matrix = tf.matmul(centered_data, centered_data, transpose_b=True)
            denom = tf.maximum(0.0, tf.cast(num_obs, dtype=tf.float32) - tf.cast(correction,dtype=tf.float32))
            cov_matrix = cov_matrix / denom
        if num_vars == 1 and num_obs == 1:
            result = tf.zeros_like(cov_matrix)
        else:
            result = cov_matrix
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float32).T
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float32).T,
        "correction": 0
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    fw_np = np.array([1, 6, 9], dtype=np.int64)
    aw_np = np.array([0.4282, 0.0255, 0.4144], dtype=np.float32)
    input_data = {
        "input": np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float32).T,
        "fweights": fw_np,
        "aweights": aw_np
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()