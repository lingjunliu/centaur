import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dims = tuple(input_dict["dims"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.tile(input_tensor, dims)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        dims = tuple(input_dict["dims"])
        
        if len(input_tensor.shape) > len(dims):
            dims = (1,) * (len(input_tensor.shape) - len(dims)) + dims
        elif len(input_tensor.shape) < len(dims):
            input_tensor = tf.reshape(input_tensor, (1,) * (len(dims) - len(input_tensor.shape)) + input_tensor.shape)
        
        result = tf.tile(input_tensor, tf.constant(dims, dtype=tf.int32))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "dims": (2, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "dims": (2,)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "dims": (2,1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    expected_shape = (2,3)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()