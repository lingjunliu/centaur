# TODO: Check tensorflow implementation, add support for upper

import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["L"])
    upper = False if "upper" not in input else input["upper"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.cholesky_inverse
    chol = torch.linalg.cholesky(input_tensor)
    inv = torch.cholesky_inverse(chol)

    if not cpu:
        inv = inv.cpu()
        
    return {"cholesky_inverse": inv.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
  
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["L"])

        # Apply TensorFlow equivalent
        chol = tf.linalg.cholesky(input_tensor)
        eye_matrix = tf.eye(tf.shape(chol)[0])
        inv = tf.linalg.cholesky_solve(chol, eye_matrix)

        return {"cholesky_inverse": inv.numpy()}

def main():
    # Example input
    input_data = {
        "L": np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)  # Symmetric positive-definite matrix
    }
    
    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure both results are compared in a common format
    torch_inv = torch_result["cholesky_inverse"]
    tf_inv = tf_result["cholesky_inverse"]
    
    if np.allclose(torch_inv, tf_inv, atol=1e-6):
        print("equal")
    else:
        print("not equal")
    
if __name__ == "__main__":
    main()