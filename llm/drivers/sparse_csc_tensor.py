import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):

    input_indices = torch.tensor(input_dict["indices"])
    input_values = torch.tensor(input_dict["values"])
    input_size = input_dict["size"]

    if not cpu:
        input_indices = input_indices.cuda()
        input_values = input_values.cuda()

    cols = input_indices[:, 1]
    rows = input_indices[:, 0]

    col_indices = torch.cat([torch.tensor([0]), torch.cumsum(torch.bincount(cols, minlength=input_size[1]), dim=0)])
    row_indices = rows
    values = input_values

    result = torch.sparse_csc_tensor(col_indices, row_indices, values, size=input_size)
    

    if not cpu:
        result = result.cpu()

    return {"result": result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.python.ops import sparse_ops

    indices = input_dict["indices"]
    values = input_dict["values"]
    dense_shape = input_dict["size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        result = tf.sparse.to_dense(sparse_tensor)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "indices": np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64),
        "values": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "size": (3, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()