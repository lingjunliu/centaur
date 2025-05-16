import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    p = input_dict.get("p", None)
    
    if not cpu:
        A = A.cuda()
    
    result = torch.linalg.cond(A, p=p)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    A_np = input_dict["A"]
    p = input_dict.get("p", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(A_np)

        if p is None:
            s = tf.linalg.svd(A, compute_uv=False)
            result = s[..., 0] / s[..., -1]
        elif p == 2:
            s = tf.linalg.svd(A, compute_uv=False)
            result = s[..., 0] / s[..., -1]
        elif p == -2:
            s = tf.linalg.svd(A, compute_uv=False)
            result = s[..., -1] / s[..., 0]
        elif p == 'fro':
            norm_A = tf.norm(A, ord= 'euclidean')
            A_inv = tf.linalg.inv(A)
            norm_A_inv = tf.norm(A_inv, ord='euclidean')
            result = norm_A * norm_A_inv
        elif p == 'nuc':
            s = tf.linalg.svd(A, compute_uv=False)
            norm_A = tf.reduce_sum(s, axis=-1)
            A_inv = tf.linalg.inv(A)
            s_inv = tf.linalg.svd(A_inv, compute_uv=False)
            norm_A_inv = tf.reduce_sum(s_inv, axis=-1)
            result = norm_A * norm_A_inv
        elif p == float('inf'):
            norm_A = tf.reduce_max(tf.reduce_sum(tf.abs(A), axis=-1), axis=-1)
            A_inv = tf.linalg.inv(A)
            norm_A_inv = tf.reduce_max(tf.reduce_sum(tf.abs(A_inv), axis=-1), axis=-1)
            result = norm_A * norm_A_inv
        elif p == float('-inf'):
            norm_A = tf.reduce_min(tf.reduce_sum(tf.abs(A), axis=-1), axis=-1)
            A_inv = tf.linalg.inv(A)
            norm_A_inv = tf.reduce_min(tf.reduce_sum(tf.abs(A_inv), axis=-1), axis=-1)
            result = norm_A * norm_A_inv
        elif p == 1:
            norm_A = tf.reduce_max(tf.reduce_sum(tf.abs(A), axis=-2), axis=-1)
            A_inv = tf.linalg.inv(A)
            norm_A_inv = tf.reduce_max(tf.reduce_sum(tf.abs(A_inv), axis=-2), axis=-1)
            result = norm_A * norm_A_inv
        elif p == -1:
            norm_A = tf.reduce_min(tf.reduce_sum(tf.abs(A), axis=-2), axis=-1)
            A_inv = tf.linalg.inv(A)
            norm_A_inv = tf.reduce_min(tf.reduce_sum(tf.abs(A_inv), axis=-2), axis=-1)
            result = norm_A * norm_A_inv
        else:
            raise ValueError(f"Unsupported norm: {p}")
            
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1., 0, -1], [0, 1, 0], [1, 0, 1]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data_fro = {
        "A": np.array([[1., 0, -1], [0, 1, 0], [1, 0, 1]], dtype=np.float32),
        "p": "fro"
    }

    torch_result_fro = torch_version(input_data_fro)
    tf_result_fro = tensorflow_version(input_data_fro)

    assert np.allclose(torch_result_fro["result"], tf_result_fro["result"], atol=A_TOL), "Results do not match"

    input_data_inf = {
        "A": np.array([[1., 0, -1], [0, 1, 0], [1, 0, 1]], dtype=np.float32),
        "p": float('inf')
    }

    torch_result_inf = torch_version(input_data_inf)
    tf_result_inf = tensorflow_version(input_data_inf)

    assert np.allclose(torch_result_inf["result"], tf_result_inf["result"], atol=A_TOL), "Results do not match"

    input_data_2 = {
        "A": np.array([[1., 0, -1], [0, 1, 0], [1, 0, 1]], dtype=np.float32),
        "p": 2
    }

    torch_result_2 = torch_version(input_data_2)
    tf_result_2 = tensorflow_version(input_data_2)

    assert np.allclose(torch_result_2["result"], tf_result_2["result"], atol=A_TOL), "Results do not match"
    
    input_data_complex = {
        "A": np.random.randn(3, 4, 4).astype(np.complex64)
    }

    torch_result_complex = torch_version(input_data_complex)
    tf_result_complex = tensorflow_version(input_data_complex)

    assert np.allclose(torch_result_complex["result"], tf_result_complex["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()