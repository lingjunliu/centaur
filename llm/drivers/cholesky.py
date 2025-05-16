import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    upper = input_dict.get("upper", False)

    if not cpu:
        A = A.cuda()

    L = torch.linalg.cholesky(A, upper=upper)

    if not cpu:
        L = L.cpu()

    return {"result": L.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        upper = input_dict.get("upper", False)

        try:
            L = tf.linalg.cholesky(A)
            if upper:
                L = tf.transpose(L)
        except tf.errors.InvalidArgumentError:
            L = np.zeros(A.shape, dtype=A.dtype.as_numpy_dtype)

        L = L.numpy()

    return {"result": L}

def main():
    A_TOL = 0.01

    A = np.array([[4.0, 1.0],
                  [1.0, 4.25]], dtype=np.float32)

    input_data = {
        "A": A
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    A = np.array([[4.0, 1.0 + 1j],
                  [1.0 - 1j, 4.25]], dtype=np.complex64)

    input_data = {
        "A": A
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    A = np.array([[4.0, 1.0],
                  [1.0, 4.25]], dtype=np.float32)

    input_data = {
        "A": A,
        "upper": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()