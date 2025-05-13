import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    crow_indices = torch.tensor(input_dict["crow_indices"])
    col_indices = torch.tensor(input_dict["col_indices"])
    values = torch.tensor(input_dict["values"])
    size = input_dict["size"]

    if not cpu:
        crow_indices = crow_indices.cuda()
        col_indices = col_indices.cuda()
        values = values.cuda()

    result = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=size)

    if not cpu:
        result = result.cpu()

    return {"result": result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    crow_indices = tf.constant(input_dict["crow_indices"], dtype=tf.int64)
    col_indices = tf.constant(input_dict["col_indices"], dtype=tf.int64)
    values = tf.constant(input_dict["values"])
    size = tf.constant(input_dict["size"], dtype=tf.int64)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        row_lengths = crow_indices[1:] - crow_indices[:-1]
        
        row_indices = tf.range(size[0])
        
        indices = []
        val_list = []
        value_idx = 0

        for i in range(size[0]):
            num_cols = row_lengths[i]
            for j in range(num_cols):
                indices.append([i, col_indices[value_idx+j]])
                val_list.append(values[value_idx+j])
            value_idx += num_cols
        
        sparse_matrix = tf.sparse.SparseTensor(
            indices=indices,
            values=val_list,
            dense_shape=size
        )

        dense_matrix = tf.sparse.to_dense(sparse_matrix)
        result = dense_matrix.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "crow_indices": np.array([0, 2, 3, 4], dtype=np.int64),
        "col_indices": np.array([0, 2, 2, 1], dtype=np.int64),
        "values": np.array([1, 2, 3, 4], dtype=np.float32),
        "size": (4, 4)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()