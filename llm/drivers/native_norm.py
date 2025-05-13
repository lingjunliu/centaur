import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 2.0)
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.linalg.norm(input_tensor, ord=p, dim=dim, keepdim=keepdim)

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
        p = input_dict.get("p", 2.0)
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)

        if dim is None:
            if p == float('inf'):
                result = tf.reduce_max(tf.abs(input_tensor))
            elif p == float('-inf'):
                result = tf.reduce_min(tf.abs(input_tensor))
            elif p == 0:
                result = tf.cast(tf.math.count_nonzero(input_tensor), tf.float32)
            else:
                result = tf.pow(tf.reduce_sum(tf.pow(tf.abs(input_tensor), p)), 1/p)

            if keepdim:
                result = tf.reshape(result, [1]*len(input_tensor.shape.as_list()))

        else:
            abs_x = tf.abs(input_tensor)
            axis = dim

            if p == float('inf'):
                result = tf.reduce_max(abs_x, axis=axis, keepdims=keepdim)
            elif p == float('-inf'):
                result = tf.reduce_min(abs_x, axis=axis, keepdims=keepdim)
            elif p == 0:
                result = tf.cast(tf.math.count_nonzero(input_tensor, axis=axis), tf.float32)
                if keepdim:
                    result = tf.expand_dims(result, axis=axis[0])
            else:
                result = tf.pow(tf.reduce_sum(tf.pow(abs_x, p), axis=axis, keepdims=keepdim), 1/p)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "p": 2,
        "dim": [1],
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32),
        "p": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "p": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "p": 0,
        "dim": [1],
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "p": float('inf'),
        "dim": [1],
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[-1., -2.], [3., 4.]], dtype=np.float32),
        "p": float('-inf'),
        "dim": [1],
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[-1., -2.], [3., 4.]], dtype=np.float32),
        "p": 1,
        "dim": [1],
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()