import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    mat1 = torch.tensor(input_dict["mat1"]).to_sparse_coo()
    mat2 = torch.tensor(input_dict["mat2"])
    
    if not cpu:
        mat1 = mat1.cuda()
        mat2 = mat2.cuda()
    
    result = torch.hspmm(mat1, mat2)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import scipy.sparse as sparse

    mat1_np = input_dict["mat1"]
    mat2_np = input_dict["mat2"]

    mat1_sparse = sparse.coo_matrix(mat1_np)
    mat1_indices = np.vstack((mat1_sparse.row, mat1_sparse.col)).T
    mat1_values = mat1_sparse.data
    mat1_shape = mat1_sparse.shape

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        mat1 = tf.sparse.SparseTensor(indices=mat1_indices, values=mat1_values, dense_shape=mat1_shape)
        mat2 = tf.constant(mat2_np)

        result = tf.sparse.sparse_dense_matmul(mat1, mat2)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "mat1": np.array([[0, 0, 1], [0, 2, 0]], dtype=np.float32),
        "mat2": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()