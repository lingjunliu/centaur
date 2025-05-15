import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices = input_dict["indices"]
    values = torch.tensor(input_dict["values"])
    accumulate = input_dict.get("accumulate", False)

    if isinstance(indices, np.ndarray):
        indices = (torch.tensor(indices),)
    elif isinstance(indices, tuple):
        indices = tuple(torch.tensor(idx) for idx in indices)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(indices, tuple):
            indices = tuple(idx.cuda() for idx in indices)
        else:
            indices = indices.cuda()
        values = values.cuda()

    input_tensor.index_put_(indices, values, accumulate=accumulate)
    
    if not cpu:
        input_tensor = input_tensor.cpu()
    
    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        indices = input_dict["indices"]
        values = tf.constant(input_dict["values"])
        accumulate = input_dict.get("accumulate", False)
        
        input_tensor_shape = tf.shape(input_tensor)
        
        if isinstance(indices, tuple):
            idx = tf.stack([tf.cast(tf.reshape(i, [-1]), tf.int32) for i in indices], axis = 1)
            updates = tf.tensor_scatter_nd_update(input_tensor, idx, tf.reshape(values, [-1]))

        else:
            updates = tf.tensor_scatter_nd_update(input_tensor, tf.expand_dims(tf.cast(tf.reshape(indices, [-1]), tf.int32), axis=1), tf.reshape(values, [-1]))

        if accumulate:
            if isinstance(indices, tuple):
              idx = tf.stack([tf.cast(tf.reshape(i, [-1]), tf.int32) for i in indices], axis = 1)
              result = tf.tensor_scatter_nd_add(input_tensor, idx, tf.reshape(values, [-1]))
            else:
              result = tf.tensor_scatter_nd_add(input_tensor, tf.expand_dims(tf.cast(tf.reshape(indices, [-1]), tf.int32), axis=1), tf.reshape(values, [-1]))
            result = tf.reshape(result, input_tensor_shape)
        else:
          result = updates

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "indices": np.array([1, 3], dtype=np.int64),
        "values": np.array([6.0, 7.0], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "indices": (np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64)),
        "values": np.array([5.0, 6.0], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "indices": np.array([1, 3], dtype=np.int64),
        "values": np.array([6.0, 7.0], dtype=np.float32),
        "accumulate": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()