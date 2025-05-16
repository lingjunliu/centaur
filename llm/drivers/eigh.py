import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    UPLO = input_dict.get("UPLO", 'L')

    if not cpu:
        A = A.cuda()

    eigenvalues, eigenvectors = torch.linalg.eigh(A, UPLO=UPLO)

    if not cpu:
        eigenvalues = eigenvalues.cpu()
        eigenvectors = eigenvectors.cpu()

    return {"eigenvalues": eigenvalues.numpy(), "eigenvectors": eigenvectors.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        UPLO = input_dict.get("UPLO", 'L')
        n = tf.shape(A)[0]

        if UPLO == 'U':
          mask = np.triu(np.ones((input_dict["A"].shape[-1], input_dict["A"].shape[-1]), dtype=bool))
          A_np = input_dict["A"]
          A_upper = np.where(mask, A_np, 0)
          A = tf.constant(A_upper)
          eigenvalues, eigenvectors = tf.linalg.eigh(A)

        else:
          mask = np.tril(np.ones((input_dict["A"].shape[-1], input_dict["A"].shape[-1]), dtype=bool))
          A_np = input_dict["A"]
          A_lower = np.where(mask, A_np, 0)
          A = tf.constant(A_lower)
          eigenvalues, eigenvectors = tf.linalg.eigh(A)

        eigenvalues = eigenvalues.numpy()
        eigenvectors = eigenvectors.numpy()

    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[2.9228+0.0000j, 0.2029-0.0862j], [0.2029+0.0862j, 0.3464+0.0000j]], dtype=np.complex128)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(torch_result["eigenvectors"], tf_result["eigenvectors"], atol=A_TOL), "Eigenvectors do not match"
    
    input_data = {
        "A": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(torch_result["eigenvectors"], tf_result["eigenvectors"], atol=A_TOL), "Eigenvectors do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64),
        "UPLO": 'U'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(torch_result["eigenvectors"], tf_result["eigenvectors"], atol=A_TOL), "Eigenvectors do not match"
    
    input_data = {
        "A": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64),
        "UPLO": 'L'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(torch_result["eigenvectors"], tf_result["eigenvectors"], atol=A_TOL), "Eigenvectors do not match"

    print("Success")

if __name__ == "__main__":
    main()