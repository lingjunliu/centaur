import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    atol = input_dict.get("atol", None)
    rtol = input_dict.get("rtol", None)
    hermitian = input_dict.get("hermitian", False)

    if not cpu:
        A = A.cuda()
        if atol is not None and isinstance(atol, np.ndarray):
            atol = torch.tensor(atol).cuda()
        if rtol is not None and isinstance(rtol, np.ndarray):
            rtol = torch.tensor(rtol).cuda()
    else:
        if atol is not None and isinstance(atol, np.ndarray):
            atol = torch.tensor(atol)
        if rtol is not None and isinstance(rtol, np.ndarray):
            rtol = torch.tensor(rtol)

    result = torch.linalg.pinv(A, atol=atol, rtol=rtol, hermitian=hermitian)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        atol = input_dict.get("atol", None)
        rtol = input_dict.get("rtol", None)
        hermitian = input_dict.get("hermitian", False)

        if hermitian:
            A = tf.cast(A, tf.complex64)
            s = tf.linalg.eigvalsh(A)
            v = tf.linalg.eigh(A)[1]
        else:
            s = tf.linalg.svd(A, compute_uv=False)
            u, _, v = tf.linalg.svd(A)
        
        if len(A.shape) == 1:
            A = tf.expand_dims(A, axis=0)

        if rtol is None:
            m, n = A.shape[-2:]
            eps = np.finfo(A.dtype.as_numpy_dtype).eps
            rtol = max(m, n) * eps
        
        if atol is None:
            atol = 0.0

        if isinstance(rtol, (int, float)):
            rtol = tf.constant(rtol, dtype=A.dtype)
        else:
            rtol = tf.constant(rtol, dtype=A.dtype)

        if isinstance(atol, (int, float)):
            atol = tf.constant(atol, dtype=A.dtype)
        else:
            atol = tf.constant(atol, dtype=A.dtype)
        
        threshold = tf.maximum(atol, tf.reduce_max(s) * rtol)
        
        mask = s > threshold
        s_masked = tf.where(mask, s, tf.zeros_like(s))
        s_inv = tf.where(mask, 1.0 / s_masked, tf.zeros_like(s_masked))
        
        if hermitian:
            s_inv_mat = tf.linalg.diag(tf.cast(s_inv, dtype=A.dtype))
            result = tf.matmul(v, tf.matmul(s_inv_mat, tf.linalg.adjoint(v)))
        else:
            s_inv_mat = tf.linalg.diag(s_inv)
            if len(u.shape) == 1:
                u = tf.reshape(u, [1, -1])
                v = tf.reshape(v, [1, -1])

            result = tf.matmul(v, tf.matmul(s_inv_mat, tf.transpose(u)))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "rtol": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "atol": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        "hermitian": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")