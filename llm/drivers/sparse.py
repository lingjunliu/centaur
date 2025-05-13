import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    indices = torch.tensor(input_dict["indices"])
    values = torch.tensor(input_dict["values"])
    size = torch.Size(input_dict["size"])
    
    if not cpu:
        indices = indices.cuda()
        values = values.cuda()

    result = torch.sparse_coo_tensor(indices, values, size).to_dense()

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    indices = input_dict["indices"]
    values = input_dict["values"]
    size = input_dict["size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        indices = tf.constant(indices, dtype=tf.int64)
        values = tf.constant(values, dtype=tf.float32)
        size = tf.constant(size, dtype=tf.int64)

        sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=size)
        result = tf.sparse.to_dense(sparse_tensor)
        result = tf.dtypes.cast(result, tf.float32)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "indices": np.array([[0, 0], [1, 2]], dtype=np.int64),
        "values": np.array([1, 2], dtype=np.float32),
        "size": (2, 5)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "indices": np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64),
        "values": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "size": (3, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "indices": np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64),
        "values": np.array([1, 2, 3], dtype=np.float32),
        "size": [2, 2]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()