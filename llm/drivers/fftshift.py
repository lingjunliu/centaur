import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.fftshift(input_tensor, dim=dim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    dim = input_dict.get("dim", None)

    if dim is None:
        axes = list(range(len(input_tensor.shape)))
    elif isinstance(dim, int):
        axes = [dim]
    else:
        axes = list(dim)

    shifted = input_tensor
    for axis in axes:
        axis_len = tf.shape(shifted)[axis]
        shift = axis_len // 2

        begin_indices = [(axis_len - shift) if i == axis else 0 for i in range(len(shifted.shape))]
        size_indices = [shift if i == axis else -1 for i in range(len(shifted.shape))]
        first_part = tf.slice(shifted, begin_indices, size_indices)

        begin_indices = [0 if i == axis else 0 for i in range(len(shifted.shape))]
        size_indices = [(axis_len - shift) if i == axis else -1 for i in range(len(shifted.shape))]
        second_part = tf.slice(shifted, begin_indices, size_indices)

        shifted = tf.concat([first_part, second_part], axis=axis)
    
    return {"result": shifted.numpy()}

def main():
    A_TOL = 0.01

    input_data_1d = {
        "input": np.array([0.0, 0.25, -0.5, -0.25], dtype=np.float32)
    }

    torch_result_1d = torch_version(input_data_1d)
    tf_result_1d = tensorflow_version(input_data_1d)
    assert np.allclose(torch_result_1d["result"], tf_result_1d["result"], atol=A_TOL), "Results do not match (1D)"
    
    input_data_2d = {
        "input": np.array([[0.0, 1.0, 2.0, -2.0, -1.0],
                           [0.1, 1.1, 2.1, -1.9, -0.9],
                           [0.2, 1.2, 2.2, -1.8, -0.8],
                           [-0.2, 0.8, 1.8, -2.2, -1.2],
                           [-0.1, 0.9, 1.9, -2.1, -1.1]], dtype=np.float32)
    }

    torch_result_2d = torch_version(input_data_2d)
    tf_result_2d = tensorflow_version(input_data_2d)
    assert np.allclose(torch_result_2d["result"], tf_result_2d["result"], atol=A_TOL), "Results do not match (2D)"
    
    input_data_2d_dim = {
        "input": np.array([[0.0, 1.0, 2.0, -2.0, -1.0],
                           [0.1, 1.1, 2.1, -1.9, -0.9],
                           [0.2, 1.2, 2.2, -1.8, -0.8],
                           [-0.2, 0.8, 1.8, -2.2, -1.2],
                           [-0.1, 0.9, 1.9, -2.1, -1.1]], dtype=np.float32),
        "dim": 1
    }

    torch_result_2d_dim = torch_version(input_data_2d_dim)
    tf_result_2d_dim = tensorflow_version(input_data_2d_dim)
    assert np.allclose(torch_result_2d_dim["result"], tf_result_2d_dim["result"], atol=A_TOL), "Results do not match (2D, dim)"

    print("Success")

if __name__ == "__main__":
    main()