import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 2.0)
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.linalg.vector_norm(input_tensor, ord=p, dim=dim, keepdim=keepdim)

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
        p = input_dict.get("p", 2.0)
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)
        
        if dim is None:
            result = tf.norm(input_tensor, ord=p, axis=None, keepdims=keepdim)
        else:
            result = tf.norm(input_tensor, ord=p, axis=dim, keepdims=keepdim)
            
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "p": 2.0,
        "dim": 1,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32),
        "p": 2.0,
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "p": 2.0,
        "dim": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()