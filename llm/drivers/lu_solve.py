import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    LU_data = torch.tensor(input_dict["LU_data"])
    LU_pivots = torch.tensor(input_dict["LU_pivots"])
    b = torch.tensor(input_dict["b"])

    if not cpu:
        LU_data = LU_data.cuda()
        LU_pivots = LU_pivots.cuda()
        b = b.cuda()

    result = torch.linalg.lu_solve(LU_data, LU_pivots, b.unsqueeze(-1)).squeeze(-1)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    LU_data_np = input_dict["LU_data"]
    LU_pivots_np = input_dict["LU_pivots"]
    b_np = input_dict["b"]
    
    LU_data = tf.constant(LU_data_np)
    LU_pivots = tf.constant(LU_pivots_np, dtype=tf.int32)
    b = tf.constant(b_np, dtype=LU_data.dtype)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"
    
    with tf.device(device_string):
        n = LU_data.shape[0]

        x = tf.identity(b)
        x = tf.cast(x, dtype=LU_data.dtype)
        x = tf.reshape(x, [-1, 1])

        permuted_x = tf.identity(x)
        for i in range(n):
          pivot_index = LU_pivots[i] - 1
          
          x_i = permuted_x[i]
          x_pivot = permuted_x[pivot_index]
          
          permuted_x = tf.tensor_scatter_nd_update(permuted_x, [[i]], [x_pivot])
          permuted_x = tf.tensor_scatter_nd_update(permuted_x, [[pivot_index]], [x_i])

        y = tf.linalg.triangular_solve(LU_data, permuted_x, lower=True)
        result = tf.linalg.triangular_solve(tf.transpose(LU_data), y, lower=False)
        result = tf.reshape(result, [-1])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "LU_data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "LU_pivots": np.array([1, 2, 3], dtype=np.int32),
        "b": np.array([10.0, 11.0, 12.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()