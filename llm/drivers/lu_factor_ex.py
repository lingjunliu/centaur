import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    pivot = input_dict.get("pivot", True)

    if not cpu:
        A = A.cuda()

    LU, pivots = torch.linalg.lu_factor(A, pivot=pivot)

    if not cpu:
        LU = LU.cpu()
        pivots = pivots.cpu()

    return {"LU": LU.numpy(), "pivots": pivots.numpy(), "infos": np.array([0])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    A_np = input_dict["A"]
    A = tf.constant(A_np)
    pivot = input_dict.get("pivot", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if A.shape[0] != A.shape[1]:
            LU, pivots = tf.linalg.qr(A)
            infos = tf.zeros((1,), dtype=tf.int32)

            return {"LU": LU.numpy(), "pivots": pivots.numpy(), "infos": infos.numpy()}

        P, L, U = tf.linalg.lu(A)
        LU = tf.matmul(L,U)

        pivots_np = np.zeros(A.shape[0], dtype=np.int32)
        P_np = P.numpy()

        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if np.allclose(A_np[i,:], LU[i,:].numpy()):
                    pivots_np[i] = j + 1
                    break
        pivots = tf.convert_to_tensor(pivots_np, dtype=tf.int32)

        infos = tf.zeros((1,), dtype=tf.int32)

        return {"LU": LU.numpy(), "pivots": pivots.numpy(), "infos": infos.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pivot": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Pivots results do not match"
    assert np.allclose(torch_result["infos"], tf_result["infos"], atol=A_TOL), "Infos results do not match"

    print("Success")

if __name__ == "__main__":
    main()