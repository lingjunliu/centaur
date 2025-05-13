import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    ord = input_dict.get("ord", None)
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    dtype = input_dict.get("dtype", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if dtype is not None:
        dtype = getattr(torch, np.dtype(dtype).name)
    
    if ord is None:
        if dtype is not None:
            result = torch.linalg.vector_norm(input_tensor, dim=dim, keepdim=keepdim, dtype=dtype)
        else:
            result = torch.linalg.vector_norm(input_tensor, dim=dim, keepdim=keepdim)
    else:
        if dtype is not None:
            result = torch.linalg.vector_norm(input_tensor, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)
        else:
            result = torch.linalg.vector_norm(input_tensor, ord=ord, dim=dim, keepdim=keepdim)
    
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
        ord = input_dict.get("ord", None)
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)
        dtype = input_dict.get("dtype", None)
        
        if dtype is not None:
            input_tensor = tf.cast(input_tensor, dtype)

        if ord is None:
            ord = 'euclidean'

        result = tf.norm(input_tensor, ord=ord, axis=dim, keepdims=keepdim)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "ord": 2,
        "dim": 1,
        "keepdim": True,
        "dtype": np.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32),
        "ord": 2,
        "dtype": np.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()