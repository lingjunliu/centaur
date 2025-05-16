import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hermitian = input_dict.get("hermitian", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.linalg.ldl_factor(input_tensor, hermitian=hermitian)
    L = torch.tril(torch.linalg.solve_triangular(result[0], torch.eye(input_tensor.shape[0], device=input_tensor.device), upper=False), diagonal=0)
    D = torch.diag(result[1])
    P = result[0]

    if not cpu:
        L = L.cpu()
        D = D.cpu()
        P = P.cpu()

    return {"L": L.numpy(), "D": D.numpy(), "P": P.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    input_tensor = tf.constant(input_dict["input"])
    hermitian = input_dict.get("hermitian", False)

    if not cpu:
        device = "/GPU:0"
    else:
        device = "/CPU:0"

    with tf.device(device):
        a = tf.cast(input_tensor, dtype=tf.float64)
        n = tf.shape(a)[0]
        p = tf.range(n)
        l = tf.eye(n, dtype=tf.float64)
        d = tf.zeros((n,), dtype=tf.float64)

        for i in tf.range(n):
            i = tf.cast(i, dtype=tf.int64)
            pivot_index = i + tf.cast(tf.argmax(tf.abs(a[i:, i])), dtype=tf.int64)
            if pivot_index != i:
                a = tf.tensor_scatter_nd_update(a, [[i], [pivot_index]], [a[[pivot_index]], a[[i]]])
                p = tf.tensor_scatter_nd_update(p, [[i], [pivot_index]], [p[[pivot_index]], p[[i]]])
                l = tf.tensor_scatter_nd_update(l, [[i], [pivot_index]], [l[[pivot_index]], l[[i]]])

            d = tf.tensor_scatter_nd_update(d, [[i]], [a[i, i]])
            range_val = n - tf.cast(i, dtype=tf.int64) - 1
            l = tf.tensor_scatter_nd_update(l, tf.stack([tf.cast(tf.range(i + 1, n), dtype=tf.int64), tf.zeros(range_val, dtype=tf.int64) + i], axis=1), a[i + 1:, i] / a[i, i])
            a = tf.tensor_scatter_nd_update(a, tf.stack([tf.cast(tf.range(i + 1, n), dtype=tf.int64), tf.cast(tf.range(i + 1, n), dtype=tf.int64)], axis=1), a[i + 1:, i + 1] - tf.reshape(l[i + 1:, i] * a[i, i], (-1,1)) * tf.transpose(tf.reshape(l[i + 1:, i], (-1,1))))

        D = tf.linalg.diag(d)
        L = tf.linalg.band_part(l, -1, 0)
        P = tf.one_hot(p, depth=n, dtype=tf.float32)

        L = tf.cast(L, dtype=tf.float32)
        D = tf.cast(D, dtype=tf.float32)
        P = tf.cast(P, dtype=tf.float32)
        
        return {"L": L.numpy(), "D": D.numpy(), "P": P.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        "hermitian": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D results do not match"
    assert np.allclose(torch_result["P"], tf_result["P"], atol=A_TOL), "P results do not match"
    
    input_data = {
        "input": np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]], dtype=np.float32),
        "hermitian": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D results do not match"
    assert np.allclose(torch_result["P"], tf_result["P"], atol=A_TOL), "P results do not match"

    print("Success")

if __name__ == "__main__":
    main()