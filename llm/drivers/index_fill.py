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
        
        index = tf.cast(index, tf.int32)
        value = tf.cast(value, input_tensor.dtype)

        if dim == 0:
            num_updates = tf.shape(index)[0]
            updates = tf.fill([num_updates, input_shape[1]], value)
            
            indices = tf.stack([index, tf.range(input_shape[1], dtype=tf.int32)], axis=1)
            indices = tf.reshape(indices, [num_updates, 2])

            update_indices = tf.stack([index, tf.range(input_shape[1], dtype=tf.int32)], axis=1)
            update_indices = tf.reshape(update_indices, [-1, 2])
            updates_reshaped = tf.tile(tf.reshape(value, [1]), [num_updates * input_shape[1]])
            updates_reshaped = tf.reshape(updates_reshaped, [num_updates, input_shape[1]])
            result = tf.tensor_scatter_nd_update(input_tensor, update_indices, tf.reshape(updates_reshaped, [-1]))

        elif dim == 1:
            num_updates = tf.shape(index)[0]
            updates = tf.fill([num_updates], value)
            indices = tf.stack([tf.zeros_like(index, dtype=tf.int32), index], axis=1)
            result = tf.tensor_scatter_nd_update(input_tensor, indices, updates)
        else:
            raise NotImplementedError(f"Only dim 0 and 1 are supported. Received: {dim}")
    
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "dim": 1,
        "index": np.array([0, 2], dtype=np.int64),
        "value": 10.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
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