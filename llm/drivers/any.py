import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if dim is None:
        result = torch.any(input_tensor)
    else:
        result = torch.any(input_tensor, dim=dim, keepdim=keepdim)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)

        input_tensor = tf.cast(input_tensor, tf.bool)

        if dim is None:
            result = tf.reduce_any(input_tensor)
        else:
            result = tf.reduce_any(input_tensor, axis=dim, keepdims=keepdim)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[True, False], [False, True]], dtype=np.bool_),
        "dim": 1,
        "keepdim": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "dim": 0,
        "keepdim": True
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    

    input_data = {
        "input": np.array([[0, 0], [0, 0]], dtype=np.int32),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 0, 1], dtype=np.int32),
        "dim": 0,
        "keepdim": False
    }
    try:
        torch_result = torch_version(input_data)
    except Exception as e:
        pass
    tf_result = tensorflow_version(input_data)
    
    input_data = {
        "input": np.array([1, 0, 1], dtype=np.int32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()