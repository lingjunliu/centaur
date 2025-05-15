import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

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

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        UPLO = input_dict.get("UPLO", 'L')

        eigenvalues, eigenvectors = tf.linalg.eigh(A)
    
    return {"eigenvalues": eigenvalues.numpy(), "eigenvectors": eigenvectors.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "A": np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32),
        "UPLO": 'L'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(np.abs(torch_result["eigenvectors"]), np.abs(tf_result["eigenvectors"]), atol=A_TOL), "Eigenvectors do not match"

    input_data = {
        "A": np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32),
        "UPLO": 'U'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(np.abs(torch_result["eigenvectors"]), np.abs(tf_result["eigenvectors"]), atol=A_TOL), "Eigenvectors do not match"

    input_data = {
        "A": np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(np.abs(torch_result["eigenvectors"]), np.abs(tf_result["eigenvectors"]), atol=A_TOL), "Eigenvectors do not match"

    print("Success")

if __name__ == "__main__":
    main()