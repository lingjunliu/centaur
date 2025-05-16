import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other = torch.tensor(input_dict["other"])
    offset = int(input_dict.get("offset", 0))
    dim1 = int(input_dict.get("dim1", 0))
    dim2 = int(input_dict.get("dim2", 1))

    if not cpu:
        input_tensor = input_tensor.cuda()
        other = other.cuda()

    torch.diagonal_copy(input_tensor, other, offset=offset, dim1=dim1, dim2=dim2)

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor_np = input_dict["input"]
    other_np = input_dict["other"]
    offset = int(input_dict.get("offset", 0))
    dim1 = int(input_dict.get("dim1", 0))
    dim2 = int(input_dict.get("dim2", 1))

    input_tensor = tf.constant(input_tensor_np)
    other = tf.constant(other_np)

    rank = len(input_tensor_np.shape)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        n = tf.shape(input_tensor)[0]
        m = tf.shape(input_tensor)[1]

        updates = tf.linalg.diag_part(other, k=offset)
        
        i = tf.range(tf.maximum(0, -offset), tf.minimum(m, n-offset))

        rows = tf.clip_by_value(i, 0, n-1)
        cols = tf.clip_by_value(i + offset, 0, m-1)
        
        
        indices = tf.stack([rows, cols], axis=1)
        
        num_valid_indices = tf.minimum(tf.shape(indices)[0], tf.shape(updates)[0])
        
        indices = indices[:num_valid_indices]
        updates = updates[:num_valid_indices]
        
        new_tensor = tf.tensor_scatter_nd_update(input_tensor, indices, updates)
        
        result = new_tensor.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "other": np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "other": np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32),
        "offset": 1
    }
    
    input_data["other"] = np.array([[0, 11, 12], [13, 0, 15], [16, 17, 0]], dtype=np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()