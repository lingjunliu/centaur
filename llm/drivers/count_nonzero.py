import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.count_nonzero(input_tensor, dim=dim)
    if keepdim:
        if dim is not None:
            result = result.reshape(list(input_tensor.shape[:dim]) + [1] + list(input_tensor.shape[dim+1:]))
        else:
            result = torch.tensor(result).reshape((1,)).repeat(input_tensor.ndim).reshape(input_tensor.shape)
            
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
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)

        result = tf.math.count_nonzero(input_tensor, axis=dim, keepdims=keepdim)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
        "dim": 1,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int32),
        "dim": 0,
        "keepdim": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int32),
        "dim": 1,
        "keepdim": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int32),
        "dim": 1,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()