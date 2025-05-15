import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.is_conj(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        if input_tensor.dtype in [tf.complex64, tf.complex128]:
            result = tf.math.reduce_any(tf.not_equal(tf.math.imag(input_tensor), 0.0))
        else:
            result = tf.constant(False)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1+1j, 2+0j, 3+2j], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.array_equal(torch_result["result"], tf_result["result"]), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.array_equal(torch_result["result"], tf_result["result"]), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()