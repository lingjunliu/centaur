import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    q = input_dict.get("q", 6)
    niter = input_dict.get("niter", 2)

    if not cpu:
        input_tensor = input_tensor.cuda()

    U, S, V = torch.svd_lowrank(input_tensor, q=q, niter=niter)

    if not cpu:
        U = U.cpu()
        S = S.cpu()
        V = V.cpu()

    return {"U": U.numpy(), "S": S.numpy(), "V": V.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    q = input_dict.get("q", 6)
    niter = input_dict.get("niter", 2)
    alpha = input_dict.get("alpha", 0.0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = input_tensor
        m = A.shape[0]
        n = A.shape[1] if len(A.shape) > 1 else 1

        if alpha == 0.0:
            S, U, V = tf.linalg.svd(A)
            S = S[:q]
            U = U[:, :q]
            V = V[:, :q]
            
        else:
            Y = A
            for _ in range(niter):
                Y = tf.matmul(A, tf.matmul(A, Y, transpose_a=True))

            Q, _ = tf.linalg.qr(Y)
            B = tf.matmul(Q, tf.matmul(A, Q, transpose_a=True))
            s, U_hat = tf.linalg.eig(B)
            s = tf.sqrt(tf.math.real(s))
            
            S = s[:q]
            U_hat = tf.math.real(U_hat)
            U_hat = U_hat[:, :q]
            U = tf.matmul(Q, U_hat)
            V = tf.matmul(A, U, transpose_a=True)
            V = tf.linalg.normalize(V, axis=0)[0]

        U = U.numpy()
        S = S.numpy()
        V = V.numpy()
    
    return {"U": U, "S": S, "V": V}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "q": 2,
        "niter": 2,
        "alpha": 0.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["U"], tf_result["U"], atol=A_TOL)
    assert np.allclose(torch_result["S"], tf_result["S"], atol=A_TOL)
    assert np.allclose(torch_result["V"], tf_result["V"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()