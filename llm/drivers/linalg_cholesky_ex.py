import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

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

    A = tf.constant(input_dict["A"])
    upper = input_dict.get("upper", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        try:
            if upper:
                A_transpose = tf.transpose(A)
                L = tf.linalg.cholesky(A_transpose)
                L = tf.transpose(L)
            else:
                L = tf.linalg.cholesky(A)
            info = tf.constant(0, dtype=tf.int32)
        except tf.errors.InvalidArgumentError:
            L = tf.zeros_like(A)
            info = tf.constant(-1, dtype=tf.int32)
        
        L = L.numpy()
        info = info.numpy()
    
    return {"L": L, "info": info}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[4.0, 12.0, -16.0],
                       [12.0, 37.0, -43.0],
                       [-16.0, -43.0, 98.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "Results do not match for L"
    assert np.array_equal(torch_result["info"], tf_result["info"]), "Results do not match for info"

    print("Success")

if __name__ == "__main__":
    main()