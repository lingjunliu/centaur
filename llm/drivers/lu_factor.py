import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    pivot = input_dict.get("pivot", True)

    if not cpu:
        A = A.cuda()
    
    LU, pivots = torch.linalg.lu_factor(A, pivot=pivot)
    info = torch.tensor(0)
    
    if not cpu:
        LU = LU.cpu()
        pivots = pivots.cpu()
        if isinstance(info, torch.Tensor):
            info = info.cpu()
    
    return {"LU": LU.numpy(), "pivots": pivots.numpy(), "info": np.array(info.item() if isinstance(info, torch.Tensor) else info)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        pivot_arg = input_dict.get("pivot", True)
        
        if pivot_arg:
            s, u, v = tf.linalg.svd(A)
            u = tf.linalg.diag(s) @ tf.transpose(v)

            P, L, U = tf.linalg.lu(A)
            L = tf.linalg.matmul(P, L)
            LU = tf.linalg.band_part(L, -1, 0) + tf.linalg.band_part(U, 0, -1) - tf.linalg.diag(tf.ones(tf.shape(A)[0], dtype=A.dtype))
            LU_np = LU.numpy()
            _, perm = tf.linalg.invert_permutation(tf.cast(tf.argmax(P, axis=0), tf.int32)).numpy()
            pivots = perm + 1
            info = 0

        else:
          P = tf.eye(tf.shape(A)[0], dtype=A.dtype)
          L = tf.linalg.band_part(A, -1, 0)
          U = tf.linalg.band_part(A, 0, -1)
          LU = L + U - tf.linalg.diag(tf.linalg.diag_part(A))
          pivots = np.arange(1, tf.shape(A)[0] + 1)
          info = 0

          LU = LU.numpy()

    return {"LU": LU, "pivots": pivots, "info": np.array(info)}

def main():
    A_TOL = 0.01
    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pivot": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Pivots do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info do not match"

    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "pivot": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Pivots do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info do not match"

    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "pivot": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Pivots do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info do not match"

    print("Success")

if __name__ == "__main__":
    main()