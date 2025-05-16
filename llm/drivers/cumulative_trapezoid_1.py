import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    y = torch.tensor(input_dict["y"])
    x = input_dict.get("x", None)
    if x is not None:
        x = torch.tensor(x)
    dim = input_dict.get("dim", -1)

    if not cpu:
        y = y.cuda()
        if x is not None:
            x = x.cuda()

    result = torch.cumulative_trapezoid(y, x=x, dim=dim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    y = tf.constant(input_dict["y"])
    x = input_dict.get("x", None)
    if x is not None:
        x = tf.constant(x)
    dim = input_dict.get("dim", -1)

    if dim < 0:
        dim = len(y.shape) + dim

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        rank = len(y.shape)
        perm = list(range(rank))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        y_perm = tf.transpose(y, perm)
        if x is not None:
            x_perm = tf.transpose(x, perm)
        else:
            x_perm = None

        if x_perm is None:
            dx = 1.0
        else:
            dx = x_perm[..., 1:] - x_perm[..., :-1]
        
        if x_perm is None:
            integral = tf.math.cumsum((y_perm[..., :-1] + y_perm[..., 1:]) * dx / 2.0, axis=-1)
        else:
            integral = tf.math.cumsum((y_perm[..., :-1] + y_perm[..., 1:]) * dx / 2.0, axis=-1)
        
        first_element = tf.zeros_like(integral[..., :1])
        result_perm = tf.concat([first_element, integral], axis=-1)
        
        result = tf.transpose(result_perm, perm)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "y": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "x": np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    min_len = min(torch_result["result"].shape[-1], tf_result["result"].shape[-1])
    torch_cropped = torch_result["result"][..., :min_len]
    tf_cropped = tf_result["result"][..., :min_len]
    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"


    input_data = {
        "y": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    min_len = min(torch_result["result"].shape[-1], tf_result["result"].shape[-1])
    torch_cropped = torch_result["result"][..., :min_len]
    tf_cropped = tf_result["result"][..., :min_len]
    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"

    input_data = {
        "y": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "x": np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    min_len = min(torch_result["result"].shape[-1], tf_result["result"].shape[-1])
    torch_cropped = torch_result["result"][..., :min_len]
    tf_cropped = tf_result["result"][..., :min_len]
    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()