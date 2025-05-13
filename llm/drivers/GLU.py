import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", -1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.glu(input_tensor, dim=dim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    dim = input_dict.get("dim", -1)

    if dim == -1:
        dim = len(input_tensor.shape) - 1

    input_shape = input_tensor.shape
    split_dim = input_shape[dim] // 2
    
    if dim == len(input_tensor.shape) - 1:
        a = input_tensor[..., :split_dim]
        b = input_tensor[..., split_dim:]
    elif dim == 0:
        a = input_tensor[:split_dim]
        b = input_tensor[split_dim:]
    else:
        slices = [slice(None)] * len(input_shape)
        slices[dim] = slice(None, split_dim)
        a = input_tensor[tuple(slices)]
        
        slices = [slice(None)] * len(input_shape)
        slices[dim] = slice(split_dim, None)
        b = input_tensor[tuple(slices)]
    
    result = a * tf.sigmoid(b)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()