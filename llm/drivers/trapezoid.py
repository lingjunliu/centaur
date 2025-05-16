import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    y = torch.tensor(input_dict["y"])
    x = input_dict.get("x", None)
    if x is not None:
        x = torch.tensor(x)
    dx = input_dict.get("dx", None)
    dim = input_dict.get("dim", -1)
    
    if not cpu:
        y = y.cuda()
        if x is not None:
            x = x.cuda()
    
    if x is not None:
        result = torch.trapezoid(y, x=x, dim=dim)
    elif dx is not None:
        result = torch.trapezoid(y, dx=dx, dim=dim)
    else:
        result = torch.trapezoid(y, dim=dim)
    
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
        y = tf.constant(input_dict["y"], dtype=tf.float32)
        x = input_dict.get("x", None)
        dx = input_dict.get("dx", None)
        dim = input_dict.get("dim", -1)

        if x is not None:
            x = tf.constant(x, dtype=tf.float32)
        
        if x is None:
            if dx is None:
                dx = 1.0
            
            result = tf.reduce_sum(dx * (y[..., 1:] + y[..., :-1]) / 2, axis=dim)
        else:
            result = tf.reduce_sum((x[..., 1:] - x[..., :-1]) * (y[..., 1:] + y[..., :-1]) / 2, axis=dim)
        
        if len(y.shape) > 1:
            if dim == 0:
                result_shape = list(y.shape)
                result_shape[0] = y.shape[0] - 1
                result = tf.reshape(result, result_shape[1:])
            elif dim == 1:
                result_shape = list(y.shape)
                result_shape[1] = y.shape[1] - 1
                result = tf.reshape(result, result_shape[0:1])
            else:
                raise ValueError("Dimension not handled")

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "y": np.array([1, 5, 10], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "y": np.array([1, 5, 10], dtype=np.float32),
        "dx": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "y": np.array([1, 5, 10], dtype=np.float32),
        "x": np.array([1, 3, 6], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "y": np.arange(9, dtype=np.float32).reshape(3, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "y": np.arange(9, dtype=np.float32).reshape(3, 3),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "y": np.ones((3, 3), dtype=np.float32),
        "x": np.array([1, 3, 6], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "y": np.ones((3, 3), dtype=np.float32),
        "x": np.array([[1, 2, 3], [1, 3, 5], [1, 4, 7]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()