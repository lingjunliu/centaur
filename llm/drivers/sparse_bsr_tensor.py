import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    indptr = torch.tensor(input_dict["indptr"])
    indices = torch.tensor(input_dict["indices"])
    values = torch.tensor(input_dict["values"])
    size = tuple(input_dict["size"])
    blocksize = tuple(input_dict["blocksize"])

    if not cpu:
        indptr = indptr.cuda()
        indices = indices.cuda()
        values = values.cuda()

    crow_indices = torch.arange(indptr.shape[0])
    result = torch.sparse_bsr_tensor(crow_indices, indices, values, size=size)

    if not cpu:
        result = result.cpu()

    return {"result": result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    indptr = input_dict["indptr"]
    indices = input_dict["indices"]
    values = input_dict["values"]
    size = input_dict["size"]
    blocksize = input_dict["blocksize"]

    num_rows = size[0] // blocksize[0]
    num_cols = size[1] // blocksize[1]
    
    dense_shape = size
    
    row_indices = []
    col_indices = []
    value_list = []

    for i in range(num_rows):
        start_index = indptr[i]
        end_index = indptr[i+1]
        
        for j in range(start_index, end_index):
            col = indices[j]
            block_values = values[j]
            
            for row_offset in range(blocksize[0]):
                for col_offset in range(blocksize[1]):
                    row = i * blocksize[0] + row_offset
                    column = col * blocksize[1] + col_offset
                    
                    value = block_values[row_offset, col_offset]
                    
                    row_indices.append(row)
                    col_indices.append(column)
                    value_list.append(value)
    
    indices_tf = tf.constant(list(zip(row_indices, col_indices)), dtype=tf.int64)
    values_tf = tf.constant(value_list, dtype=values.dtype)

    sparse_tensor = tf.sparse.SparseTensor(indices_tf, values_tf, dense_shape=dense_shape)
    
    result = tf.sparse.to_dense(sparse_tensor).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "indptr": np.array([0, 1, 2], dtype=np.int64),
        "indices": np.array([0, 1], dtype=np.int64),
        "values": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        "size": (4, 4),
        "blocksize": (2, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()