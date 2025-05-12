import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    result = torch.equal(input_tensor, other_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        other_tensor = tf.constant(input_dict["other"])
        
        size_equal = tf.reduce_all(tf.equal(tf.shape(input_tensor), tf.shape(other_tensor)))
        
        if not size_equal:
            result = tf.constant(False)
        else:
            elements_equal = tf.reduce_all(tf.equal(input_tensor, other_tensor))
        
            if tf.reduce_any(tf.math.is_nan(tf.cast(input_tensor, tf.float32))) or tf.reduce_any(tf.math.is_nan(tf.cast(other_tensor, tf.float32))):
                result = tf.constant(False)
            else:
                result = tf.logical_and(size_equal, elements_equal)
        
        result = result.numpy()
    
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "other": np.array([1, 2, 3], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "other": np.array([1, 2, 4], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "other": np.array([1, 2], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([3, np.nan], dtype=np.float32),
        "other": np.array([3, np.nan], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()