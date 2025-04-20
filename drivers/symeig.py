import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    eigenvectors = input.get("eigenvectors", False)
    upper = input.get("upper", True)

    UPLO = 'U' if upper else 'L'

    # Apply to torch.linalg.eigh
    eigvals, eigvecs = torch.linalg.eigh(input_tensor, UPLO=UPLO)

    if not cpu:
        eigvals = eigvals.cpu()
        eigvecs = eigvecs.cpu()

    return {
        "symeig_eigenvalues": eigvals.numpy(),
        "symeig_eigenvectors": eigvecs.numpy() if eigenvectors else None
    }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Note: TensorFlow does not support an 'upper' parameter directly,
        # tf.linalg.eigh automatically assumes a lower triangular part of the matrix.
        input_tensor = tf.linalg.band_part(input_tensor, -1, 0) if not input.get("upper", True) else input_tensor

        # Apply to TensorFlow equivalent
        eigvals, eigvecs = tf.linalg.eigh(input_tensor)

        return {
            "symeig_eigenvalues": eigvals.numpy(),
            "symeig_eigenvectors": eigvecs.numpy() if input.get("eigenvectors", False) else None
        }

def compare_eigenvectors(torch_eigenvectors, tf_eigenvectors):
    for i in range(torch_eigenvectors.shape[1]):
        if np.allclose(torch_eigenvectors[:, i], tf_eigenvectors[:, i], rtol=1e-5, atol=1e-6):
            continue
        elif np.allclose(torch_eigenvectors[:, i], -tf_eigenvectors[:, i], rtol=1e-5, atol=1e-6):
            continue
        else:
            return False
    return True

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float32),  # Ensure matrix is symmetric
        "eigenvectors": True,
        "upper": True
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure reproducibility and comparison
    np.testing.assert_allclose(torch_result["symeig_eigenvalues"], tf_result["symeig_eigenvalues"], rtol=1e-5, atol=1e-6)

    if input_data["eigenvectors"]:
        eigenvector_comparison = compare_eigenvectors(torch_result["symeig_eigenvectors"], tf_result["symeig_eigenvectors"])
        if not eigenvector_comparison:
            raise AssertionError("Eigenvectors do not match within the acceptable tolerance levels.")
    
    print("Comparison result: equal")

if __name__ == "__main__":
    main()