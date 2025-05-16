import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    input_np = input_dict["input"]
    if input_np.dtype == np.float16:
        input_np = input_np.astype(np.float32)
    elif input_np.dtype == np.dtype('bfloat16'):
        input_np = input_np.astype(np.float32)

    dtype = torch.tensor(input_np).dtype
    
    if not cpu:
        torch.set_default_device("cuda")

    result = torch.finfo(dtype)
    
    if not cpu:
        torch.set_default_device("cpu")
    
    return {"result": np.array([result.bits, result.eps, result.max, result.min, result.tiny])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from ml_dtypes import bfloat16

    input_np = input_dict["input"]
    if input_np.dtype == np.float16:
        input_np = input_np.astype(np.float32)
    elif input_np.dtype == bfloat16:
        input_np = input_np.astype(np.float32)

    dtype = tf.constant(input_np).dtype
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if dtype == tf.float16:
            finfo = tf.experimental.numpy.finfo(tf.float16.as_numpy_dtype)
            eps = finfo.eps
            tiny = finfo.tiny
            max_val = finfo.max
            min_val = finfo.min
            bits = 16
        elif dtype == tf.float32:
            finfo = tf.experimental.numpy.finfo(tf.float32.as_numpy_dtype)
            eps = finfo.eps
            tiny = finfo.tiny
            max_val = finfo.max
            min_val = finfo.min
            bits = 32
        elif dtype == tf.float64:
            finfo = tf.experimental.numpy.finfo(tf.float64.as_numpy_dtype)
            eps = finfo.eps
            tiny = finfo.tiny
            max_val = finfo.max
            min_val = finfo.min
            bits = 64
        elif dtype == tf.bfloat16:
            finfo = tf.experimental.numpy.finfo(tf.bfloat16.as_numpy_dtype)
            eps = finfo.eps
            tiny = finfo.tiny
            max_val = finfo.max
            min_val = finfo.min
            bits = 16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        result = np.array([bits, eps, max_val, min_val, tiny])
    
    return {"result": result}

def main():
    A_TOL = 0.01
    from ml_dtypes import bfloat16
    # Example input
    input_data = {
        "input": np.array([1.0], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0], dtype=np.float64),
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0], dtype=np.float16),
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=1), "Results do not match"

    input_data = {
        "input": np.array([1.0], dtype=bfloat16),
    }

    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=0.1), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()