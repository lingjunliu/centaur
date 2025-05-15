import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    out = input_dict.get("out", None)
    
    if not cpu:
        A = A.cuda()
    
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
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
        
        eigenvalues, eigenvectors = tf.linalg.eig(A)
        
    return {"eigenvalues": eigenvalues.numpy(), "eigenvectors": eigenvectors.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["eigenvalues"], tf_result["eigenvalues"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(torch_result["eigenvectors"], tf_result["eigenvectors"], atol=A_TOL), "Eigenvectors do not match"

    print("Success")

if __name__ == "__main__":
    main()