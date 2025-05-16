import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dims = input_dict["dims"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.empty_permuted(tuple(input_tensor.tolist()), dims)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor_np = input_dict["input"]
    dims = input_dict["dims"]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_shape = input_tensor_np.shape
        if len(input_shape) == 0:
            permuted_shape = np.array(input_tensor_np).reshape(1).transpose(dims).shape
        else:
            permuted_shape = np.array(input_tensor_np).transpose(dims).shape

        result = tf.zeros(permuted_shape, dtype=input_tensor_np.dtype)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([2, 3, 5], dtype=np.int32),
        "dims": [2, 0, 1]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].shape, tf_result["result"].shape, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()