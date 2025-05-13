import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    other = torch.tensor(input_dict["other"])
    dim = input_dict["dim"]
    index = torch.tensor(input_dict["index"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other = other.cuda()
        index = index.cuda()

    result = input_tensor.clone()
    
    if dim == 0:
        result[index] = other
    elif dim == 1:
        for i, idx in enumerate(index):
            result[:, idx] = other[i]
    elif dim == 2:
      for i, idx in enumerate(index):
        result[:,:,idx] = other[i]
    else:
        raise ValueError("Dimension not supported")

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    other = tf.constant(input_dict["other"])
    dim = input_dict["dim"]
    index = input_dict["index"]
    
    if dim == 0:
        indices = tf.reshape(index, [-1, 1])
        updates = other
    elif dim == 1:
        num_updates = tf.shape(index)[0]
        indices = tf.stack([tf.zeros(num_updates, dtype=tf.int64), index], axis=1)
        updates = other
        indices = tf.reshape(indices, [num_updates, 2])

    else:
        raise ValueError("Dimension not supported")
    
    result = tf.tensor_scatter_nd_update(input_tensor, tf.cast(indices, tf.int32), updates)

    return {"result": result.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "other": np.array([[10, 11, 12]], dtype=np.float32),
        "dim": 0,
        "index": np.array([1], dtype=np.int64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "other": np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32),
        "dim": 0,
        "index": np.array([0,1], dtype=np.int64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "other": np.array([[10, 11], [12, 13], [14, 15]], dtype=np.float32),
        "dim": 1,
        "index": np.array([0, 1], dtype=np.int64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")