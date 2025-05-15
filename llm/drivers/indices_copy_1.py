import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    source = torch.tensor(input_dict["source"])
    accumulate = input_dict.get("accumulate", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        source = source.cuda()

    if len(input_tensor.shape) == 1:
        input_tensor.index_copy_(0, index, source)
    else:
        for i in range(input_tensor.shape[1]):
            input_tensor[:,i].index_copy_(0, index[:,i], source[:,i])

    result = input_tensor

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
        index = tf.constant(input_dict["index"])
        source = tf.constant(input_dict["source"])
        accumulate = input_dict.get("accumulate", False)

        input_tensor_shape = tf.shape(input_tensor)
        index_shape = tf.shape(index)
        source_shape = tf.shape(source)

        rank = tf.rank(input_tensor)
        
        if rank == 1:
            indices = tf.expand_dims(index, axis=1)
            updates = source

            if accumulate:
                result = tf.tensor_scatter_nd_add(input_tensor, indices, updates)
            else:
                result = tf.tensor_scatter_nd_update(input_tensor, indices, updates)
        elif rank == 2:
            ind = tf.transpose(index)
            indices = tf.stack([ind[0], ind[1]], axis = 1)
            updates = tf.reshape(source, [-1])
            if accumulate:
                result = tf.tensor_scatter_nd_add(input_tensor, tf.cast(indices, dtype=tf.int32), updates)
            else:
                result = tf.tensor_scatter_nd_update(input_tensor, tf.cast(indices, dtype=tf.int32), updates)

        result = result.numpy()
    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "index": np.array([0, 2, 4], dtype=np.int64),
        "source": np.array([10, 20, 30], dtype=np.float32),
        "accumulate": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
        "source": np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=np.float32),
        "accumulate": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()