import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dims = input_dict["dims"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    original_shape = list(input_tensor.shape)
    permuted_shape = [original_shape[i] if i < len(original_shape) else 1 for i in dims]
    result = torch.empty(permuted_shape, dtype=input_tensor.dtype, layout=input_tensor.layout, device=input_tensor.device, pin_memory=input_tensor.is_pinned())

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
        dims = input_dict["dims"]

        original_shape = tf.shape(input_tensor)
        rank = len(dims)
        perm = list(range(rank))
        for i in range(rank):
            perm[i] = dims[i] if dims[i] < len(original_shape) else 0

        target_shape = [original_shape[i] if i < len(original_shape) else 1 for i in dims]

        result = tf.zeros(target_shape, dtype=input_tensor.dtype)

        rank = len(dims)
        perm_tf = list(range(rank))
        for i in range(rank):
            perm_tf[i] = dims[i] if dims[i] < len(original_shape) else 0

        result = tf.transpose(result, perm=perm_tf)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([2, 3, 4], dtype=np.int32),
        "dims": [2, 0, 1]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"].shape, tf_result["result"].shape, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()