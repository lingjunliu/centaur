import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    # Unpack inputs from dictionary
    crow_indices = torch.tensor(input_dict["crow_indices"])
    col_indices = torch.tensor(input_dict["col_indices"])
    values = torch.tensor(input_dict["values"])
    shape = input_dict["shape"]
    
    if not cpu:
        crow_indices = crow_indices.cuda()
        col_indices = col_indices.cuda()
        values = values.cuda()
    
    # Perform torch addition
    try:
        result = torch.sparse_bsc_tensor(crow_indices, col_indices, values, size=shape)
    except RuntimeError as e:
        print(f"Caught RuntimeError: {e}")
        return {"result": np.zeros(shape)}
    
    # Move result to CPU for consistent return format
    if not cpu:
        result = result.cpu()
    
    return {"result": result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # Unpack inputs from dictionary
        crow_indices = input_dict["crow_indices"]
        col_indices = input_dict["col_indices"]
        values = tf.constant(input_dict["values"])
        shape = input_dict["shape"]
        
        indices_list = []
        row_index = 0
        for i in range(len(crow_indices)-1):
            start = crow_indices[i]
            end = crow_indices[i+1]
            for j in range(start, end):
                indices_list.append([row_index, col_indices[j]])
            row_index += 1
        
        #handle the last row if necessary
        if len(crow_indices) > 1:
            start = crow_indices[-1]
            for j in range(start, len(col_indices)):
                indices_list.append([row_index, col_indices[j]])

        if len(indices_list) == 0:
            indices = tf.constant([], dtype=tf.int64, shape=(0,2))
        else:
            indices = tf.constant(indices_list, dtype=tf.int64)
        

        sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape=shape)
        result = tf.sparse.to_dense(sparse_tensor)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "crow_indices": np.array([0, 2, 3], dtype=np.int64),
        "col_indices": np.array([0, 1, 2], dtype=np.int64),
        "values": np.array([1, 2, 3], dtype=np.float32),
        "shape": (3, 3)
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)

    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()