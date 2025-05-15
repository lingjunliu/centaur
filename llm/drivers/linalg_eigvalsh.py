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

    A = tf.constant(input_dict["A"])
    UPLO = input_dict.get("UPLO", 'L')

    if not cpu:
        device = '/GPU:0'
    else:
        device = '/CPU:0'

    with tf.device(device):
        if UPLO == 'L':
            A_sym = tf.linalg.band_part(A, -1, 0)
        else:
            A_sym = tf.linalg.band_part(A, 0, -1)

        if len(A.shape) == 3:
            result = tf.map_fn(lambda x: tf.linalg.eigvalsh(x), A_sym)
        else:
            result = tf.linalg.eigvalsh(A_sym)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        "UPLO": 'U'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "A": np.array([[[1.0, 2.0], [2.0, 1.0]],[[3.0, 4.0],[4.0, 3.0]]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()