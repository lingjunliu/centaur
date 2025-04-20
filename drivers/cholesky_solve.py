import numpy as np

def torch_cholesky_solve_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    A = torch.tensor(input["A"])
    L = torch.linalg.cholesky(A)
    B = torch.tensor(input["B"])
    upper = input.get("upper", False)
    
    # Apply torch.cholesky_solve
    X = torch.cholesky_solve(B, L, upper=upper)
    
    if not cpu:
        X = X.cpu()

    return {"cholesky_solve_result": X.numpy()}

def tensorflow_cholesky_solve_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        A = tf.constant(input["A"])
        L = tf.linalg.cholesky(A)
        B = tf.constant(input["B"])
        upper = input.get("upper", False)
        
        # Solve using TensorFlow equivalents
        if upper:
            raise ValueError("TensorFlow does not directly support 'upper' argument for cholesky_solve; convert matrix format before use.")
        
        X = tf.linalg.triangular_solve(L, B)
        X = tf.linalg.triangular_solve(tf.transpose(L), X, lower=False)
        
        return {"cholesky_solve_result": X.numpy()}

def main():
    # Example input
    input_data = {
        "A": np.array([[4.0, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 3.0]], dtype=np.float32),
        "B": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "upper": False  # For simplicity, we assume lower triangle by default
    }

    # Torch example
    torch_result = torch_cholesky_solve_version(input_data)
    print("Torch result:", torch_result["cholesky_solve_result"])

    # TensorFlow example
    tf_result = tensorflow_cholesky_solve_version(input_data)
    print("TensorFlow result:", tf_result["cholesky_solve_result"])

    # Assert and print result
    if np.allclose(torch_result["cholesky_solve_result"], tf_result["cholesky_solve_result"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()