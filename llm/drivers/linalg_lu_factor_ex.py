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

    if not cpu:
        LU = LU.cpu()
        pivots = pivots.cpu()

    return {"LU": LU.numpy(), "pivots": pivots.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    A_np = input_dict["A"]
    A = tf.constant(A_np)
    pivot = input_dict.get("pivot", True)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):
        lu, perm = tf.linalg.lu(tf.cast(A, tf.float64))
        lu = tf.cast(lu, A.dtype)

        pivots_np = np.argsort(np.argsort(perm.numpy())) + 1
        
        return {"LU": lu.numpy(), "pivots": pivots_np.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pivot": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"]), "Pivots results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "pivot": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["LU"], tf_result["LU"], atol=A_TOL), "LU results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"]), "Pivots results do not match"

    print("Success")

if __name__ == "__main__":
    main()