import numpy as np

def torch_mean(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    keepdim = input.get("keepdim", False)

    if "dtype" in input.keys():
        dtype = torch.tensor(np.array([], dtype=input["dtype"])).dtype
    else:
        dtype = None

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.mean
    if dim is None:
        if dtype is None:
            result = torch.mean(input_tensor)
        else:
            result = torch.mean(input_tensor, dtype=dtype)
    else:
        if dtype is None:
            result = torch.mean(input_tensor, dim=dim, keepdim=keepdim)
        else:
            result = torch.mean(input_tensor, dim=dim, keepdim=keepdim, dtype=dtype)
    if not cpu:
        result = result.cpu()

    return {"mean": result.numpy()}

def tensorflow_mean(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", None)
        keepdim = input.get("keepdim", False)
        if "dtype" in input.keys():
            dtype = tf.as_dtype(input["dtype"])
        else:
            dtype = None
        
        # Apply tf.reduce_mean
        if dim is None:
            result = tf.reduce_mean(input_tensor, dtype=dtype) if dtype else tf.reduce_mean(input_tensor)
        else:
            result = tf.reduce_mean(input_tensor, axis=dim, keepdims=keepdim, dtype=dtype) if dtype else tf.reduce_mean(input_tensor, axis=dim, keepdims=keepdim)

        result = result.numpy()  # Convert to numpy array

        return {"mean": result}

def main():
    # Case 1: Mean of all elements in the input tensor
    input_data_all_elements = {
        "input": np.random.randn(4, 3).astype(np.float32),  # Random 4x3 tensor
        "dim": None,
        "keepdim": False
    }

    # Case 2: Mean of elements along a specific dimension
    input_data_along_dim = {
        "input": np.random.randn(4, 3).astype(np.float32),  # Random 4x3 tensor
        "dim": 1,  # Mean along the columns
        "keepdim": True
    }

    # Torch and TensorFlow examples for Case 1
    torch_result_all_elements = torch_mean(input_data_all_elements)
    print("Torch result (all elements):", torch_result_all_elements)
    tf_result_all_elements = tensorflow_mean(input_data_all_elements)
    print("TensorFlow result (all elements):", tf_result_all_elements)

    assert np.allclose(torch_result_all_elements["mean"], tf_result_all_elements["mean"]), 'not equal'
    print('equal')

    # Torch and TensorFlow examples for Case 2
    torch_result_along_dim = torch_mean(input_data_along_dim)
    print("Torch result (along dim):", torch_result_along_dim)
    tf_result_along_dim = tensorflow_mean(input_data_along_dim)
    print("TensorFlow result (along dim):", tf_result_along_dim)

    assert np.allclose(torch_result_along_dim["mean"], tf_result_along_dim["mean"]), 'not equal'
    print('equal')

if __name__ == "__main__":
    main()