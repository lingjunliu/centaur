import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    input_tensor = input_tensor.type(torch.complex64)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.view_as_real(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    input_tensor = tf.cast(input_tensor, tf.complex64)

    real = tf.math.real(input_tensor)
    imag = tf.math.imag(input_tensor)
    real_imag = tf.stack([real, imag], axis=-1)

    return {"result": real_imag.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()