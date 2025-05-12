import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    start_dim = input_dict.get("start_dim", 0)
    end_dim = input_dict.get("end_dim", -1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.flatten(input_tensor, start_dim=start_dim, end_dim=end_dim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    start_dim = input_dict.get("start_dim", 0)
    end_dim = input_dict.get("end_dim", -1)
    
    rank = len(input_tensor.shape)

    if end_dim == -1:
        end_dim = rank - 1

    shape = tf.shape(input_tensor)
    
    prefix_shape = shape[:start_dim]
    suffix_shape = shape[end_dim+1:]
    
    if start_dim > end_dim:
        flattened_shape = shape
    else:
        flattened_dim_size = tf.reduce_prod(shape[start_dim:end_dim+1])
        flattened_shape = tf.concat([prefix_shape, [flattened_dim_size], suffix_shape], axis=0)
        
    result = tf.reshape(input_tensor, flattened_shape)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        "start_dim": 1
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        "start_dim": 1,
        "end_dim": 2
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        "start_dim": 0,
        "end_dim": 1
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()