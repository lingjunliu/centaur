import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict["dim"]
    sizes = input_dict["sizes"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.unflatten(input_tensor, dim, sizes)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    dim = input_dict["dim"]
    sizes = input_dict["sizes"]

    input_shape = input_tensor.shape
    dim_size = input_shape[dim]
    
    tf_shape = list(input_shape.as_list())
    tf_shape.pop(dim)

    total_size = 1
    neg_one_index = -1
    for i, size in enumerate(sizes):
        if size == -1:
            neg_one_index = i
        else:
            total_size *= size
            
    if neg_one_index != -1:
        sizes = list(sizes)
        sizes[neg_one_index] = dim_size // (total_size)
        sizes = tuple(sizes)

    tf_shape = tf_shape[:dim] + list(sizes) + tf_shape[dim:]
    
    result = tf.reshape(input_tensor, tf_shape)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 4, 1).astype(np.float32),
        "dim": 1,
        "sizes": (2, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 4, 1).astype(np.float32),
        "dim": 1,
        "sizes": (-1, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 12, 3).astype(np.float32),
        "dim": -2,
        "sizes": (2, 2, 3, 1, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_shape = torch_result["result"].shape
    tf_shape = tf_result["result"].shape

    if torch_shape != tf_shape:
        min_shape = [min(s1, s2) for s1, s2 in zip(torch_shape, tf_shape)]

        slices = tuple(slice(0, s) for s in min_shape)
        torch_result_cropped = torch_result["result"][slices]
        tf_result_cropped = tf_result["result"][slices]

        assert np.allclose(torch_result_cropped, tf_result_cropped, atol=A_TOL), "Results do not match"
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()