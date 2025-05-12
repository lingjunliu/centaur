import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    if input_tensor.dtype == torch.int32 or input_tensor.dtype == torch.int64:
        input_tensor = input_tensor.float()

    result = torch.ldexp_(input_tensor, other_tensor)

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
        other_tensor = tf.constant(input_dict["other"])
        
        input_tensor_dtype = input_tensor.dtype
        other_tensor_dtype = other_tensor.dtype
        
        if input_tensor_dtype != tf.float32 and input_tensor_dtype != tf.float64:
            input_tensor = tf.cast(input_tensor, tf.float32)
            other_tensor = tf.cast(other_tensor, tf.int32)
            power_tensor = tf.cast(other_tensor, tf.float32)
        elif input_tensor_dtype == tf.float64:
            other_tensor = tf.cast(other_tensor, tf.int32)
            power_tensor = tf.cast(other_tensor, tf.float64)
        else:
            other_tensor = tf.cast(other_tensor, tf.int32)
            power_tensor = tf.cast(other_tensor, tf.float32)
            
        result = input_tensor * tf.pow(tf.cast(2.0, dtype=input_tensor.dtype), power_tensor)
        
        if input_tensor_dtype != tf.float32 and input_tensor_dtype != tf.float64:
            result = tf.cast(result, input_tensor_dtype)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "other": np.array([2, 3, 4], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "other": np.array([2, 3, 4], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "other": np.array([2, 3, 4], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()