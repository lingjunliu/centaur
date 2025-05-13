import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    n_slices = input_dict["n_slices"]
    dim = input_dict.get("dim", 0)
    step = input_dict.get("step", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    slices = torch.chunk(input_tensor, n_slices, dim=dim)
    
    if not cpu:
        slices = [s.cpu() for s in slices]

    return {"result": [s.numpy() for s in slices]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    n_slices = input_dict["n_slices"]
    dim = input_dict.get("dim", 0)
    step = input_dict.get("step", 1)

    shape = tf.shape(input_tensor)
    dim_size = shape[dim]

    slice_size = dim_size // n_slices

    if step != 1:
      raise NotImplementedError("TensorFlow version does not support step != 1")

    slices = []
    for i in range(n_slices):
        start = [0] * len(input_tensor.shape)
        start[dim] = i * slice_size
        size = tf.unstack(shape)
        size[dim] = slice_size
        size = tf.stack(size)

        slice_tensor = tf.slice(input_tensor, start, size)
        slices.append(slice_tensor)
    
    return {"result": [s.numpy() for s in slices]}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32),
        "n_slices": 2,
        "dim": 0,
        "step": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_slice, tf_slice in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_slice, tf_slice, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32),
        "n_slices": 2,
        "dim": 1,
        "step": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_slice, tf_slice in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_slice, tf_slice, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()