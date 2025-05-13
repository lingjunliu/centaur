import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"]).to_sparse()
    mat = torch.tensor(input_dict["mat"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        mat = mat.cuda()

    result = torch.smm(input_tensor, mat)

    if not cpu:
        result = result.cpu()

    return {"result": result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor_np = input_dict["input"]
        mat_np = input_dict["mat"]

        indices = np.array(input_tensor_np["indices"], dtype=np.int64)
        values = np.array(input_tensor_np["values"], dtype=np.float32)
        dense_shape = np.array(input_tensor_np["dense_shape"], dtype=np.int64)

        input_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        mat = tf.constant(mat_np, dtype=np.float32)

        result = tf.sparse.sparse_dense_matmul(input_tensor, mat)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": {
            "indices": [[0, 0], [1, 2]],
            "values": [1, 2],
            "dense_shape": [2, 3]
        },
        "mat": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    }

    input_data_torch = {
        "input": np.array([[1,0,0],[0,0,2]], dtype=np.float32),
        "mat": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    }

    torch_result = torch_version(input_data_torch)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()