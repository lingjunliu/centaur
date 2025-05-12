import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    other = input_dict["other"]
    if isinstance(other, float) or isinstance(other, int):
        other_tensor = other
    else:
        other_tensor = torch.tensor(other)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if not isinstance(other, float) and not isinstance(other, int):
            other_tensor = other_tensor.cuda()
    
    result = torch.fmod(input_tensor, other_tensor)
    
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
        other = input_dict["other"]

        if isinstance(other, float) or isinstance(other, int):
            other_tensor = tf.constant(other)
        else:
            other_tensor = tf.constant(other)

        input_tensor = tf.cast(input_tensor, tf.float32)
        other_tensor = tf.cast(other_tensor, tf.float32)

        result = input_tensor - tf.floor(tf.divide(input_tensor, other_tensor)) * other_tensor
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-3., -2, -1, 1, 2, 3], dtype=np.float32),
        "other": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "other": -1.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "other": np.array([2, 1, 3, 1, 2], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "other": -1.5
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()