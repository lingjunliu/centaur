import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict["dim"]
    index = input_dict["index"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.select(input_tensor, dim, index)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    dim = input_dict["dim"]
    index = input_dict["index"]

    rank = len(input_tensor.shape)
    indices = []
    updates = []

    if dim == 0:
        result = input_tensor[index]
    elif dim == 1:
        result = input_tensor[:, index]
    elif dim == 2:
        result = input_tensor[:, :, index]
    elif dim == 3:
        result = input_tensor[:, :, :, index]
    else:
        selection = [slice(None)] * rank
        selection[dim] = index
        result = input_tensor[tuple(selection)]
    

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 4, 5).astype(np.float32),
        "dim": 2,
        "index": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 4).astype(np.float32),
        "dim": 0,
        "index": 1
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 4, 5, 6).astype(np.float32),
        "dim": 1,
        "index": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()