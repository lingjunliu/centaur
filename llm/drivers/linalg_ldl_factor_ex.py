import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):

    A = torch.tensor(input_dict["A"])
    pivot = input_dict.get("pivot", False)

    if not cpu:
        A = A.cuda()

    L, D, pivots = torch.linalg.ldl_factor_ex(A, hermitian=False, check_errors=False)
    if pivot:
        P = torch.eye(A.shape[0])
        for i in range(len(pivots)):
            if pivots[i] != i + 1:
                P[[i, pivots[i]-1], :] = P[[pivots[i]-1, i], :]
        L = P @ L
    

    if not cpu:
        L = L.cpu()
        D = D.cpu()
        pivots = torch.tensor(pivots).cpu()

    return {"L": L.numpy(), "D": D.numpy(), "pivots": pivots.numpy()}

def tensorflow_version(input_dict, cpu=True):

    A_np = input_dict["A"]
    pivot = input_dict.get("pivot", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(A_np, dtype=tf.float32)
        n = A.shape[0]
        dtype = A.dtype

        L = tf.eye(n, dtype=dtype)
        D = tf.zeros((n, n), dtype=dtype)
        pivots = np.arange(n)

        if pivot:
            P = np.eye(n)
            A_np_copy = A_np.copy()
            pivots = np.arange(n)
            for i in range(n):
                pivot_index = i + np.argmax(np.abs(A_np_copy[i:, i]))
                if pivot_index != i:
                    A_np_copy[[i, pivot_index], :] = A_np_copy[[pivot_index, i], :]
                    P[[i, pivot_index], :] = P[[pivot_index, i], :]
                    pivots[[i, pivot_index]] = pivots[[pivot_index, i]]
            A = tf.constant(A_np_copy, dtype=tf.float32)
            P = tf.constant(P, dtype=tf.float32)
            L = tf.linalg.matmul(P, L)

        for i in range(n):
            D = tf.tensor_scatter_nd_update(D, [[i, i]], [A[i, i]])
            L = tf.tensor_scatter_nd_update(L, [[i, i]], [1.0])
            if i < n - 1:
                L_updates = A[i+1:,i:i+1] / A[i, i]
                indices = tf.constant([[j, i] for j in range(i + 1, n)])
                L = tf.tensor_scatter_nd_update(L, indices, tf.reshape(L_updates, [-1]))
                A_updates = A[i+1:,i:] - tf.expand_dims(A[i+1:,i], axis=1) * tf.expand_dims(A[i,i:], axis=0) / A[i, i]
                indices = tf.constant([[j, k] for j in range(i + 1, n) for k in range(i, n)])
                A = tf.tensor_scatter_nd_update(A, indices, tf.reshape(A_updates, [-1]))

        L = L.numpy()
        D = D.numpy()
        pivots = np.array(pivots)
    
    return {"L": L, "D": D, "pivots": pivots}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "A": np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=np.float32),
        "pivot": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L Results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D Results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Pivots Results do not match"

    input_data = {
        "A": np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=np.float32),
        "pivot": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L Results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D Results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Pivots Results do not match"


    print("Success")

if __name__ == "__main__":
    main()