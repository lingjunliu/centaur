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
    
    if dim < 0:
        dim = len(input_shape) + dim

    split_size = input_shape[dim] // 2
    
    input_list = tf.split(input_tensor, num_or_size_splits=[split_size, split_size], axis=dim)
    
    result = input_list[0] * tf.sigmoid(input_list[1])

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()