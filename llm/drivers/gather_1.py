import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict["dim"]
    index = torch.tensor(input_dict["index"])
    sparse_grad = input_dict.get("sparse_grad", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()

    result = torch.gather(input_tensor, dim, index, sparse_grad=sparse_grad)

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
        dim = input_dict["dim"]
        index = tf.constant(input_dict["index"])
        
        input_shape = input_tensor.shape.as_list()
        index_shape = index.shape.as_list()
        output_shape = index_shape

        rank = len(input_shape)
        
        if rank == 1:
            result = tf.gather(input_tensor, index)
        elif rank == 2:
            if dim == 0:
                row_indices = tf.tile(tf.expand_dims(tf.range(index_shape[1]), 0), [index_shape[0], 1])
                stacked = tf.stack([index, tf.cast(row_indices, tf.int64)], axis=-1)
                result = tf.gather_nd(input_tensor, stacked)
            elif dim == 1:
                col_indices = tf.tile(tf.expand_dims(tf.range(index_shape[0]), 1), [1, index_shape[1]])
                stacked = tf.stack([tf.cast(col_indices, tf.int64), index], axis=-1)
                result = tf.gather_nd(input_tensor, stacked)
            else:
                raise ValueError("Dimension must be 0 or 1 for 2D tensors")
        elif rank == 3:
            if dim == 0:
                batch_range = tf.range(index_shape[0])
                row_range = tf.range(index_shape[1])
                col_range = tf.range(index_shape[2])

                batch_tiled = tf.tile(tf.expand_dims(tf.expand_dims(batch_range, 1), 2), [1, index_shape[1], index_shape[2]])
                row_tiled = tf.tile(tf.expand_dims(tf.expand_dims(row_range, 0), 2), [index_shape[0], 1, index_shape[2]])
                col_tiled = tf.tile(tf.expand_dims(tf.expand_dims(col_range, 0), 0), [index_shape[0], index_shape[1], 1])
                
                stacked = tf.stack([index, tf.cast(row_tiled, tf.int64), tf.cast(col_tiled, tf.int64)], axis=-1)
                result = tf.gather_nd(input_tensor, stacked)
            elif dim == 1:
                batch_range = tf.range(index_shape[0])
                row_range = tf.range(index_shape[1])
                col_range = tf.range(index_shape[2])

                batch_tiled = tf.tile(tf.expand_dims(tf.expand_dims(batch_range, 1), 2), [1, index_shape[1], index_shape[2]])
                row_tiled = tf.tile(tf.expand_dims(tf.expand_dims(row_range, 0), 2), [index_shape[0], 1, index_shape[2]])
                col_tiled = tf.tile(tf.expand_dims(tf.expand_dims(col_range, 0), 0), [index_shape[0], index_shape[1], 1])
                
                stacked = tf.stack([tf.cast(batch_tiled, tf.int64), index, tf.cast(col_tiled, tf.int64)], axis=-1)

                result = tf.gather_nd(input_tensor, stacked)
            elif dim == 2:
                 batch_range = tf.range(index_shape[0])
                 row_range = tf.range(index_shape[1])
                 col_range = tf.range(index_shape[2])

                 batch_tiled = tf.tile(tf.expand_dims(tf.expand_dims(batch_range, 1), 2), [1, index_shape[1], index_shape[2]])
                 row_tiled = tf.tile(tf.expand_dims(tf.expand_dims(row_range, 0), 2), [index_shape[0], 1, index_shape[2]])
                 col_tiled = tf.tile(tf.expand_dims(tf.expand_dims(col_range, 0), 0), [index_shape[0], index_shape[1], 1])

                 stacked = tf.stack([tf.cast(batch_tiled, tf.int64), tf.cast(row_tiled, tf.int64), index], axis=-1)
                 result = tf.gather_nd(input_tensor, stacked)
            else:
                raise ValueError("Dimension must be 0, 1, or 2 for 3D tensors")
        else:
            raise ValueError("Tensor rank must be 1, 2 or 3")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "dim": 1,
        "index": np.array([[0, 0], [1, 0]], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "dim": 0,
        "index": np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], dtype=np.int64)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()