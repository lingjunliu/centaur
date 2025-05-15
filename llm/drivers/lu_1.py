import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    pivot = input_dict.get("pivot", True)
    get_infos = input_dict.get("get_infos", False)
    
    if not cpu:
        A = A.cuda()
    
    if get_infos:
        A_LU, pivots, info = torch.lu(A, pivot=pivot, get_infos=get_infos)
        if not cpu:
            A_LU = A_LU.cpu()
            pivots = pivots.cpu()
            info = info.cpu()
        return {"A_LU": A_LU.numpy(), "pivots": pivots.numpy(), "info": info.numpy()}
    else:
        A_LU, pivots = torch.lu(A, pivot=pivot)
        if not cpu:
            A_LU = A_LU.cpu()
            pivots = pivots.cpu()
        return {"A_LU": A_LU.numpy(), "pivots": pivots.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    A = tf.constant(input_dict["A"])
    pivot = input_dict.get("pivot", True)
    get_infos = input_dict.get("get_infos", False)
    
    if not cpu:
        device = "/GPU:0"
    else:
        device = "/CPU:0"
        
    with tf.device(device):
        try:
            P, L, U = tf.linalg.lu(A)
            pivots = tf.argmax(P, axis=0) + 1
            s = tf.cast(tf.shape(A)[0], tf.float32)
            L = tf.linalg.set_diag(L, tf.ones(tf.shape(A)[0]) )

            LU = tf.matmul(L,U)
            if get_infos:
                info = tf.zeros((), dtype=tf.int32)
                return {"A_LU": LU.numpy(), "pivots": pivots.numpy(), "info": info.numpy()}
            else:
                return {"A_LU": LU.numpy(), "pivots": pivots.numpy()}
        except tf.errors.InvalidArgumentError as e:
            if "Input is not invertible" in str(e):
                print("Matrix is not invertible in tensorflow, returning zero matrices")
                shape = A.shape
                lu = tf.zeros(shape)
                pivots = tf.zeros(shape[0], dtype=tf.int64)
                if get_infos:
                    info = tf.constant(1, dtype=tf.int32)
                    return {"A_LU": lu.numpy(), "pivots": pivots.numpy(), "info": info.numpy()}
                else:
                    return {"A_LU": lu.numpy(), "pivots": pivots.numpy()}
            else:
                raise e

def main():
    A_TOL = 0.01
    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "pivot": True,
        "get_infos": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["A_LU"], tf_result["A_LU"], atol=A_TOL), "Results do not match for A_LU"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Results do not match for pivots"
    
    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "pivot": True,
        "get_infos": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["A_LU"], tf_result["A_LU"], atol=A_TOL), "Results do not match for A_LU when get_infos is True"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Results do not match for pivots when get_infos is True"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Results do not match for info when get_infos is True"
    
    
    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "pivot": True,
        "get_infos": True
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["A_LU"], tf_result["A_LU"], atol=A_TOL), "Results do not match for A_LU when get_infos is True"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Results do not match for pivots when get_infos is True"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Results do not match for info when get_infos is True"
    

    print("Success")

if __name__ == "__main__":
    main()