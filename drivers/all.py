import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if dim is None:
        result = torch.all(input_tensor)
    else:
        result = torch.all(input_tensor, dim=dim, keepdim=keepdim)
    
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
        input_tensor = tf.cast(input_tensor, tf.bool)
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)

        if dim is None:
            result = tf.reduce_all(input_tensor)
        else:
            result = tf.reduce_all(input_tensor, axis=dim, keepdims=keepdim)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_no_dim = {
        "input": np.array([True, True, True], dtype=bool)
    }

    input_data_with_dim = {
        "input": np.array([[True, True], [True, False]], dtype=bool),
        "dim": 1,
        "keepdim": False
    }

    torch_result_no_dim = torch_version(input_data_no_dim)
    tf_result_no_dim = tensorflow_version(input_data_no_dim)
    assert np.allclose(torch_result_no_dim["result"], tf_result_no_dim["result"], atol=A_TOL), "Results do not match"

    torch_result_with_dim = torch_version(input_data_with_dim)
    tf_result_with_dim = tensorflow_version(input_data_with_dim)
    assert np.allclose(torch_result_with_dim["result"], tf_result_with_dim["result"], atol=A_TOL), "Results do not match"
    
    input_data_int = {
        "input": np.array([0, 1, 2])
    }
    
    torch_result_int = torch_version(input_data_int)
    tf_result_int = tensorflow_version(input_data_int)
    
    assert np.allclose(torch_result_int["result"], tf_result_int["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()