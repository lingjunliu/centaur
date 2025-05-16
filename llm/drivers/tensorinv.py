import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    ind = input_dict.get("ind", 2)

    if not cpu:
        A = A.cuda()

    result = torch.linalg.tensorinv(A, ind=ind)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    A = tf.constant(input_dict["A"])
    ind = input_dict.get("ind", 2)

    A_shape = A.shape
    A_rank = len(A_shape)
    
    m_dims = A_shape[:ind]
    n_dims = A_shape[ind:]

    m = tf.reduce_prod(m_dims)
    n = tf.reduce_prod(n_dims)

    if m != n:
        raise RuntimeError("The reshaped A is not invertible or the product of the first ind dimensions is not equal to the product of the rest.")

    A_reshaped = tf.reshape(A, (m, n))
    
    try:
        A_inv_reshaped = tf.linalg.inv(A_reshaped)
    except tf.errors.InvalidArgumentError:
        raise RuntimeError("The reshaped A is not invertible or the product of the first ind dimensions is not equal to the product of the rest.")
    
    output_shape = A_shape[ind:] + A_shape[:ind]
    A_inv = tf.reshape(A_inv_reshaped, output_shape)
    
    return {"result": A_inv.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.eye(4 * 6).reshape((4, 6, 8, 3)).astype(np.float32),
        "ind": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.random.rand(4, 4).astype(np.float32),
        "ind": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()