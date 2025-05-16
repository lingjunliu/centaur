import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    source = torch.tensor(input_dict["source"])
    dim = input_dict.get("dim", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        source = source.cuda()

    result = torch.index_copy(input_tensor, dim, index, source)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    index = tf.constant(input_dict["index"])
    source = tf.constant(input_dict["source"])
    dim = input_dict.get("dim", 0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        index_shape = tf.shape(index)
        source_shape = tf.shape(source)

        num_indices = tf.shape(index)[0]

        rank = len(input_tensor.shape)

        updates = []
        indices = []

        for i in range(num_indices):
            idx = index[i]
            
            if rank == 1:
              indices.append(idx)
              updates.append(source[i])
            elif rank == 2:
              rep = [1] * rank
              rep[dim] = input_shape[dim]
              
              temp_indices = tf.reshape(tf.range(input_shape[1 - dim]), [1, -1])
              
              if dim == 0:
                stacked_indices = tf.stack([tf.tile(tf.expand_dims(tf.cast(idx, dtype=tf.int64), axis=0), [tf.shape(temp_indices)[1]]), tf.cast(temp_indices[0], dtype=tf.int64)], axis = 1)
              else:
                stacked_indices = tf.stack([tf.cast(temp_indices[0], dtype=tf.int64), tf.tile(tf.expand_dims(tf.cast(idx, dtype=tf.int64), axis=0), [tf.shape(temp_indices)[1]])], axis = 1)
              
              for j in range(input_shape[1-dim]):
                indices.append(stacked_indices[j])
                updates.append(source[i, j])
            else:
              raise Exception("Rank > 2 not implemented")

        if rank == 1:
            result = tf.tensor_scatter_nd_update(tf.identity(input_tensor), tf.expand_dims(tf.cast(indices, dtype=tf.int32), axis=1), updates)
        elif rank == 2:
          result = tf.tensor_scatter_nd_update(tf.identity(input_tensor), tf.cast(indices, dtype=tf.int32), updates)
        else:
            raise Exception("Rank > 2 not implemented")
            
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([10.0, 11.0], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()