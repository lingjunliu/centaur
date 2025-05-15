import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    full_matrices = input_dict.get("full_matrices", True)
    driver = input_dict.get("driver", None)

    if not cpu:
        A = A.cuda()

    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices, driver=driver)

    if not cpu:
        U = U.cpu()
        S = S.cpu()
        Vh = Vh.cpu()

    return {"U": U.numpy(), "S": S.numpy(), "Vh": Vh.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        full_matrices = input_dict.get("full_matrices", True)
        
        S, U, V = tf.linalg.svd(A, full_matrices=full_matrices)
        
        V = tf.linalg.adjoint(V)

        if not full_matrices:
             k = min(A.shape[0], A.shape[1])
             U = U[:, :k]
             V = V[:, :k]
        
        Vh = tf.linalg.adjoint(V)
    
        return {"U": U.numpy(), "S": S.numpy(), "Vh": Vh.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "A": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "full_matrices": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["U"], tf_result["U"], atol=A_TOL), "U results do not match"
    assert np.allclose(torch_result["S"], tf_result["S"], atol=A_TOL), "S results do not match"
    assert np.allclose(torch_result["Vh"], tf_result["Vh"], atol=A_TOL), "Vh results do not match"

    print("Success")

if __name__ == "__main__":
    main()