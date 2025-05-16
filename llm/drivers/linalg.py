import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict.get("A"))
    b = torch.tensor(input_dict.get("b"))
    atol = input_dict.get("atol", None)
    rtol = input_dict.get("rtol", None)
    
    if not cpu:
        A = A.cuda()
        b = b.cuda()
    
    result = torch.linalg.solve(A, b)
    
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
        A = tf.constant(input_dict.get("A"))
        b = tf.constant(input_dict.get("b"))
        
        A_shape = A.shape
        b_shape = b.shape
        
        if len(A_shape) == 0 or len(b_shape) == 0:
            raise ValueError("A and b must be tensors with rank >= 1")

        if len(b_shape) == 1:
            b = tf.reshape(b, (-1, 1))

        result = tf.linalg.solve(A, b)

        if len(b_shape) == 1:
            result = tf.reshape(result, (-1,))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "b": np.array([5, 11], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.float32),
        "b": np.array([1, 2, 3], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.float32),
        "b": np.array([[1], [2], [3]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()