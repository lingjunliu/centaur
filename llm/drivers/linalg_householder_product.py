import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    v = torch.tensor(input_dict["v"])
    
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    v = v.unsqueeze(0).unsqueeze(0)
    v = v.transpose(-1, -2)

    if not cpu:
        input_tensor = input_tensor.cuda()
        v = v.cuda()
    
    result = torch.linalg.householder_product(input_tensor, v)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        v = tf.constant(input_dict["v"], dtype=tf.float32)

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        v = tf.expand_dims(v, axis=0)
        v = tf.transpose(v, perm=[0, 2, 1])

        v = tf.cast(v, input_tensor.dtype)

        norm_squared = tf.reduce_sum(tf.square(v))
        
        if norm_squared == 0:
           result = input_tensor
        else:
           projection = tf.reduce_sum(input_tensor * v, axis=1, keepdims=True) / norm_squared
           result = input_tensor - 2 * projection * v

        result = tf.squeeze(result)
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "v": np.array([0.5, 0.5, 0.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "v": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "v": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()