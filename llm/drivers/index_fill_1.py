import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict["dim"]
    index = torch.tensor(input_dict["index"])
    value = torch.tensor(input_dict["value"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        value = value.cuda()
    
    result = torch.index_fill(input_tensor, dim, index, value)

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
        input_tensor = tf.constant(input_dict["input"])
        dim = input_dict["dim"]
        index = tf.constant(input_dict["index"])
        value = tf.constant(input_dict["value"])

        input_shape = tf.shape(input_tensor)
        num_indices = tf.shape(index)[0]

        if tf.rank(input_tensor) == 1:
            updates = tf.fill([num_indices], value)
            indices = tf.expand_dims(index, axis=1)
        else:
            if dim == 0:
                updates = tf.fill([num_indices] + [input_shape[i] for i in range(1, len(input_shape))], value)
                indices = tf.stack([index, tf.zeros(num_indices, dtype=tf.int64)], axis=1)
            else:
                updates = tf.fill([num_indices], value)
                if num_indices > 0:
                    row_indices = tf.range(tf.shape(input_tensor)[0])
                    row_indices = tf.repeat(row_indices, 1)
                    
                    col_indices = index
                    
                    indices = tf.stack([tf.zeros_like(col_indices, dtype=tf.int64)+ row_indices[0], col_indices], axis=1)
                else:
                    indices = tf.constant([], dtype=tf.int64, shape=(0,2))


        result = tf.tensor_scatter_nd_update(input_tensor, indices, updates)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 2], dtype=np.int64),
        "value": 10.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "dim": 1,
        "index": np.array([0, 2], dtype=np.int64),
        "value": 10.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 2], dtype=np.int64),
        "value": 10.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()