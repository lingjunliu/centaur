import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    UPLO = input_dict.get("UPLO", 'L')

    if not cpu:
        A = A.cuda()

    result = torch.linalg.eigvalsh(A, UPLO=UPLO)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        UPLO = input_dict.get("UPLO", 'L')

        if UPLO == 'L':
            A_tri = tf.linalg.band_part(A, -1, 0)
        else:
            A_tri = tf.linalg.band_part(A, 0, -1)
        
        if tf.as_dtype(A.dtype).is_complex:
           A_tri = tf.cast(A_tri, dtype=tf.complex128)
           eigvals = tf.math.real(tf.linalg.eigvalsh(A_tri))
        else:
           eigvals = tf.linalg.eigvalsh(A_tri)

        result = eigvals.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[2.9228, 0.2029], [0.2029, 0.3464]], dtype=np.float64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[2.9228, 0.2029], [0.2029, 0.3464]], dtype=np.float64),
        "UPLO": 'U'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "A": np.array([[[ 2.5797, 3.4629], [3.4629, 1.3780]],[[-4.1605, 1.3780],[1.3780, -3.1113]], [[-3.1113, 2.7381], [2.7381, 2.5797]]], dtype=np.float64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[2.9228+0.0000j, 0.2029-0.0862j], [0.2029+0.0862j, 0.3464+0.0000j]], dtype=np.complex128),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[[ 2.5797+0.0000j, 3.4629-0.0000j], [3.4629+0.0000j, 1.3780+0.0000j]],[[-4.1605+0.0000j, 1.3780-0.0000j],[1.3780+0.0000j, -3.1113-0.0000j]], [[-3.1113+0.0000j, 2.7381-0.0000j], [2.7381+0.0000j, 2.5797+0.0000j]]], dtype=np.complex128),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()