import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    other = torch.tensor(input_dict["other"])

    if not cpu:
        input = input.cuda()
        index = index.cuda()
        other = other.cuda()

    result = input.clone()
    for i in range(index.shape[0]):
        result[:, index[i]] = other[:, i]

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
        index_tensor = tf.constant(input_dict["index"])
        other_tensor = tf.constant(input_dict["other"])

        input_shape = input_tensor.shape
        num_cols = input_shape[1]

        result = tf.identity(input_tensor)
        for i in range(index_tensor.shape[0]):
            col_index = tf.cast(index_tensor[i], tf.int32)
            other_col = other_tensor[:, i]
            
            indices = tf.stack([tf.range(tf.shape(input_tensor)[0]), tf.repeat(col_index, tf.shape(input_tensor)[0])], axis=1)

            updates = other_col
            
            result = tf.tensor_scatter_nd_update(result, indices, updates)


        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "index": np.array([1, 0], dtype=np.int64),
        "other": np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()