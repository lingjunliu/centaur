import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])

    if not cpu:
        A = A.cuda()

    result = torch.linalg.Tensor(A).adjoint().resolve_conj()

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
        result = tf.linalg.adjoint(tf.cast(A, dtype=tf.complex64))
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0+1.j, 2.0+2.j], [3.0+3.j, 4.0+4.j]], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()