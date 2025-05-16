import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    other = torch.tensor(input_dict["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        other = other.cuda()

    for i, col_idx in enumerate(index):
        input_tensor[:, col_idx] = other[:, i]

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        index = tf.cast(input_dict["index"], dtype=tf.int32)
        other = tf.constant(input_dict["other"])
        
        num_rows = tf.shape(input_tensor)[0]
        num_cols = tf.shape(input_tensor)[1]
        
        row_indices = tf.range(num_rows)
        
        result = tf.identity(input_tensor)
        
        for i in tf.range(tf.shape(index)[0]):
            col_idx = index[i]
            updates = other[:, i]
            
            indices = tf.stack([row_indices, tf.cast(tf.fill([num_rows], col_idx), dtype=tf.int32)], axis=1)
            
            result = tf.tensor_scatter_nd_update(result, indices, updates)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "other": np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float32),
        "index": np.array([1, 3, 0], dtype=np.int64),
        "other": np.array([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()