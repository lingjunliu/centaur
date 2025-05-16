import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    pivot = input_dict.get("pivot", True)

    if not cpu:
        A = A.cuda()

    P, L, U = torch.linalg.lu(A, pivot=pivot)

    if not cpu:
        P = P.cpu()
        L = L.cpu()
        U = U.cpu()

    return {"P": P.numpy(), "L": L.numpy(), "U": U.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    A = tf.constant(input_dict["A"])
    pivot = input_dict.get("pivot", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        lu, p = tf.linalg.lu(A)
        num_rows = tf.shape(A)[0]
        num_cols = tf.shape(A)[1]
        L = tf.linalg.band_part(lu, num_rows - 1, 0)
        L = tf.linalg.set_diag(L, tf.ones(min(num_rows, num_cols), dtype=A.dtype))
        U = tf.linalg.band_part(lu, 0, -1)
        P = tf.eye(num_rows, dtype=A.dtype)
        if pivot:
            P = tf.gather(P, p)

        P = P.numpy()
        L = L.numpy()
        U = U.numpy()

    return {"P": P, "L": L, "U": U}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["P"], tf_result["P"], atol=A_TOL)
    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL)
    assert np.allclose(torch_result["U"], tf_result["U"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()