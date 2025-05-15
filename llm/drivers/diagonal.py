import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    offset = input_dict.get("offset", 0)
    dim1 = input_dict.get("dim1", -2)
    dim2 = input_dict.get("dim2", -1)
    
    if not cpu:
        A = A.cuda()
    
    result = torch.linalg.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    A = tf.constant(input_dict["A"])
    offset = input_dict.get("offset", 0)
    dim1 = input_dict.get("dim1", -2)
    dim2 = input_dict.get("dim2", -1)

    rank = len(A.shape)
    dim1 = dim1 if dim1 >= 0 else rank + dim1
    dim2 = dim2 if dim2 >= 0 else rank + dim2

    if dim1 > dim2:
        dim1, dim2 = dim2, dim1

    result = tf.linalg.diag_part(A, k=offset)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    
    input_data = {
        "A": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "offset": 0,
        "dim1": 0,
        "dim2": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "A": np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32),
        "offset": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()