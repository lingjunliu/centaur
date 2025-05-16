import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    values = torch.tensor(input_dict["values"])
    indices = input_dict["indices"]
    if isinstance(indices, list):
        indices = tuple(torch.tensor(idx) for idx in indices)
    else:
        indices = torch.tensor(indices)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        values = values.cuda()
        if isinstance(indices, tuple):
            indices = tuple(idx.cuda() for idx in indices)
        else:
            indices = indices.cuda()

    result = torch.as_strided_scatter(input_tensor, values, indices)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        values = tf.constant(input_dict["values"])
        indices = input_dict["indices"]

        if isinstance(indices, list):
            indices = tuple(tf.constant(idx) for idx in indices)
        else:
            indices = tf.constant(indices)

        input_shape = tf.shape(input_tensor)
        values_shape = tf.shape(values)
        
        if isinstance(indices, tuple):
            num_index_dims = len(indices)
            
            strides = []
            sizes = []
            
            updates = values
            
            for i in range(num_index_dims):
                strides.append(tf.ones_like(indices[i], dtype=tf.int32))
                sizes.append(tf.shape(indices[i])[0])
                
            if num_index_dims > 1:
                multi_indices = tf.stack(indices, axis=-1)
                flat_indices = tf.linalg.matvec(tf.ones((tf.shape(multi_indices)[0], num_index_dims), dtype=tf.int32), tf.cast(input_shape, dtype=tf.int32)[:num_index_dims])
                flat_updates = values
                
                updates = tf.tensor_scatter_nd_update(tf.zeros(input_shape, dtype=values.dtype), tf.expand_dims(tf.cast(flat_indices, dtype=tf.int64), axis=1), flat_updates)
                result = tf.add(input_tensor, updates)
            else:
                updates = tf.tensor_scatter_nd_update(tf.zeros(input_shape, dtype=values.dtype), tf.expand_dims(tf.cast(indices, dtype=tf.int64), axis=1), values)
                result = tf.add(input_tensor, updates)

        else:
            updates = tf.tensor_scatter_nd_update(tf.zeros(input_shape, dtype=values.dtype), tf.expand_dims(tf.cast(indices, dtype=tf.int64), axis=1), values)
            result = tf.add(input_tensor, updates)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.zeros((5, 5), dtype=np.float32),
        "values": np.array([1, 2, 3], dtype=np.float32),
        "indices": ([0, 2, 4], [0, 2, 4])
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.zeros((5, 5), dtype=np.float32),
        "values": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "indices": np.array([0,1,2,3,4], dtype=np.int64)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.zeros((5, 5, 5), dtype=np.float32),
        "values": np.array([1, 2, 3], dtype=np.float32),
        "indices": ([0, 2, 4], [0, 2, 4], [0, 2, 4])
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()