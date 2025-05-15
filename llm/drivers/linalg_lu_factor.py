import numpy as np
import tensorflow as tf

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

    return {"LU": LU.numpy(), "pivots": pivots.numpy()}

def tensorflow_version(input_dict, cpu=True):

    A_np = input_dict["A"]
    A = tf.constant(A_np)
    pivot = input_dict.get("pivot", True)

    if not cpu:
        device = "/GPU:0"
    else:
        device = "/CPU:0"

    with tf.device(device):
        lu, piv = tf.linalg.lu(A)

        LU = lu.numpy()
        
        num_rows = A_np.shape[0]
        pivots_np = np.zeros(num_rows, dtype=np.int32)

        for i in range(num_rows):
            for j in range(num_rows):
                if abs(piv[i, j] - 1) < 1e-6:
                    pivots_np[i] = j + 1
                    break

        pivots = pivots_np.astype(np.int32)

    return {"LU": LU, "pivots": pivots}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"]), "Pivots results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"]), "Pivots results do not match"

    print("Success")

if __name__ == "__main__":
    main()