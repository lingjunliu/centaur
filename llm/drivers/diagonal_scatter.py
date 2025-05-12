import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    values = torch.tensor(input_dict["values"])
    offset = input_dict.get("offset", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        values = values.cuda()

    result = torch.diagonal_scatter(input_tensor, values, offset=offset)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor_np = input_dict["input"]
    values_np = input_dict["values"]
    offset = input_dict.get("offset", 0)

    input_tensor = tf.constant(input_tensor_np)
    values = tf.constant(values_np)

    rank = len(input_tensor_np.shape)

    shape = tf.shape(input_tensor)
    new_tensor = tf.identity(input_tensor)
    
    num_rows = shape[-2]
    num_cols = shape[-1]
    
    diag_len = tf.minimum(num_rows, num_cols + offset) if offset >= 0 else tf.minimum(num_rows - abs(offset), num_cols)
    diag_len = tf.minimum(diag_len, tf.shape(values)[0])
    
    for i in tf.range(diag_len):
        row_index = i
        col_index = i + tf.maximum(-offset, 0)
        
        indices_list = []
        for _ in range(rank - 2):
            indices_list.append(0)
        indices_list.extend([row_index, col_index])
        indices = tf.constant([indices_list], dtype=tf.int32)

        updates = values[i]

        new_tensor = tf.tensor_scatter_nd_update(new_tensor, tf.expand_dims(indices, axis=0), tf.expand_dims(updates, axis=0))
    
    return {"result": new_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.zeros((3, 3), dtype=np.float32),
        "values": np.array([1, 2, 3], dtype=np.float32),
        "offset": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.zeros((4, 4), dtype=np.float32),
        "values": np.array([1, 2, 3, 4], dtype=np.float32),
        "offset": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.zeros((5, 5), dtype=np.float32),
        "values": np.array([1, 2, 3, 4], dtype=np.float32),
        "offset": -1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()