import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"]).long()
    other = torch.tensor(input_dict["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()
        other = other.cuda()

    result = torch.index_copy(input_tensor, 1, indices, other)

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
        indices = tf.constant(input_dict["indices"])
        other = tf.constant(input_dict["other"])

        input_shape = tf.shape(input_tensor)
        updates_shape = tf.shape(other)
        indices_shape = tf.shape(indices)

        num_updates = tf.shape(indices)[0]
        num_rows = tf.shape(input_tensor)[0]
        num_cols = tf.shape(input_tensor)[1]
        
        row_indices = tf.range(num_rows)
        row_indices = tf.expand_dims(row_indices, axis=1)
        row_indices = tf.tile(row_indices, [1, num_updates])
        row_indices = tf.reshape(row_indices, [-1])

        col_indices = tf.tile(indices, [num_rows])
        
        update_indices = tf.stack([row_indices, col_indices], axis=1)
        
        updates = tf.reshape(other, [-1])
        
        result = tf.tensor_scatter_nd_update(input_tensor, update_indices, updates)

        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "indices": np.array([0, 2], dtype=np.int32),
        "other": np.array([[7, 8], [9, 10]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()