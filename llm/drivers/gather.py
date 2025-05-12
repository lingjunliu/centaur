import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict["dim"]
    index = torch.tensor(input_dict["index"], dtype=torch.int64)
    sparse_grad = input_dict.get("sparse_grad", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()

    result = torch.gather(input_tensor, dim, index, sparse_grad=sparse_grad)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    dim = input_dict["dim"]
    index = tf.constant(input_dict["index"], dtype=tf.int32)

    if not cpu:
        device = "/GPU:0"
    else:
        device = "/CPU:0"

    with tf.device(device):
        result = tf.gather(input_tensor, index, axis=dim)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "dim": 1,
        "index": np.array([[0, 0], [1, 0]], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "dim": 2,
        "index": np.array([[[0, 0], [1, 0]], [[1, 0], [0, 1]]], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "dim": 0,
        "index": np.array([[[0, 0], [1, 0]], [[1, 0], [0, 1]]], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()