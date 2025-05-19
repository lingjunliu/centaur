import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    upper = input_dict.get("upper", False)

    if not cpu:
        A = A.cuda()

    L, info = torch.linalg.cholesky_ex(A, upper=upper)

    if not cpu:
        L = L.cpu()
        info = info.cpu()

    return {"L": L.numpy(), "info": info.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    A = tf.constant(input_dict["A"])
    upper = input_dict.get("upper", False)

    if upper:
        A = tf.transpose(A)

    try:
        L = tf.linalg.cholesky(A)
        info = tf.constant(0, dtype=tf.int32)
    except tf.errors.InvalidArgumentError:
        L = tf.zeros_like(A)
        info = tf.constant(1, dtype=tf.int32)
        

    if not upper:
      mask = tf.linalg.band_part(tf.ones_like(A), -1, 0)
      L = L * mask
    else:
      mask = tf.linalg.band_part(tf.ones_like(A), 0, -1)
      L = L * mask


    if upper:
        L = tf.transpose(L)
    
    return {"L": L.numpy(), "info": info.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[4.0, 12.0, -16.0],
                       [12.0, 37.0, -43.0],
                       [-16.0, -43.0, 17.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL, equal_nan=True), "Results do not match L"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Results do not match info"

    input_data = {
        "A": np.array([[4.0, 12.0, -16.0],
                       [12.0, 37.0, -43.0],
                       [-16.0, -43.0, 17.0]], dtype=np.float32),
        "upper": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL, equal_nan=True), "Results do not match L"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Results do not match info"

    input_data = {
        "A": np.array([[1.0, 2.0],
                       [2.0, 1.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Results do not match info"

    print("Success")


if __name__ == "__main__":
    main()