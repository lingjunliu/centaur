import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    hermitian = input_dict.get("hermitian", False)

    if not cpu:
        A = A.cuda()

    result = torch.linalg.ldl_factor(A, hermitian=hermitian)
    L = result[0]
    D = result[1]
    pivots = result[2] if len(result) > 2 else torch.arange(A.shape[0])

    if not cpu:
        L = L.cpu()
        D = D.cpu()
        pivots = pivots.cpu()

    return {"L": L.numpy(), "D": D.numpy(), "pivots": pivots.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    A_np = input_dict["A"]
    hermitian = input_dict.get("hermitian", False)
    A = tf.constant(A_np, dtype=tf.float64)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):

        n = A.shape[0]

        def _tf_ldl_factor(A, hermitian=False):

            n = tf.shape(A)[0]

            L = tf.eye(n, dtype=tf.float64)
            D = tf.zeros([n], dtype=tf.float64)
            pivots = tf.range(n)

            for k in tf.range(n):

                max_val = tf.abs(A[k, k])
                max_index = k

                for i in tf.range(k + 1, n):
                    if tf.abs(A[i, k]) > max_val:
                        max_val = tf.abs(A[i, k])
                        max_index = i

                def swap(L, D, pivots, k, max_index):
                    L_k_temp = tf.identity(L[k, :k])
                    L_max_temp = tf.identity(L[max_index, :k])
                    
                    indices_k = tf.stack([tf.tile([k], [tf.minimum(k,n)]), tf.range(tf.minimum(k,n))], axis=1)
                    indices_max = tf.stack([tf.tile([max_index], [tf.minimum(k,n)]), tf.range(tf.minimum(k,n))], axis=1)
                    
                    L = tf.tensor_scatter_nd_update(L, tf.cast(indices_k,tf.int32), L_max_temp[:tf.minimum(k,n)])
                    L = tf.tensor_scatter_nd_update(L, tf.cast(indices_max, tf.int32), L_k_temp[:tf.minimum(k,n)])

                    pivots_k_temp = tf.identity(pivots[k])
                    pivots_max_temp = tf.identity(pivots[max_index])
                    pivots = tf.tensor_scatter_nd_update(pivots, [[k]], [pivots_max_temp])
                    pivots = tf.tensor_scatter_nd_update(pivots, [[max_index]], [pivots_k_temp])
                    return L, D, pivots

                L, D, pivots = tf.cond(tf.not_equal(k, max_index), lambda: swap(L, D, pivots, k, max_index), lambda: (L, D, pivots))

                pivots = tf.cast(pivots, dtype=tf.int32)
                perm = tf.one_hot(pivots, depth=n, dtype=tf.float64)
                A = tf.matmul(tf.matmul(perm, A), tf.transpose(perm))

                D = tf.tensor_scatter_nd_update(D, [[k]], [A[k, k]])

                for i in tf.range(k + 1, n):
                    L = tf.tensor_scatter_nd_update(L, [[i, k]], [A[i, k] / D[k]])
                    for j in tf.range(k + 1, n):
                        A = tf.tensor_scatter_nd_update(A, [[i, j]], [A[i, j] - L[i, k] * L[j, k] * D[k]])
            L = tf.linalg.band_part(L, -1, 0)
            return L, D, pivots

        L, D, pivots = _tf_ldl_factor(A, hermitian=hermitian)
        L = L.numpy()
        D = D.numpy()
        pivots = pivots.numpy()

    return {"L": L, "D": D, "pivots": pivots}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=np.float32),
        "hermitian": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "pivots results do not match"

    input_data = {
        "A": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "hermitian": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "pivots results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        "hermitian": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "pivots results do not match"

    print("Success")

if __name__ == "__main__":
    main()