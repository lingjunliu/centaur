import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)

    window_length = torch.tensor(window_length)
    periodic = torch.tensor(periodic)
    
    if not cpu:
        window_length = window_length.cuda()
        if isinstance(periodic, torch.Tensor):
            periodic = periodic.cuda()
    
    result = torch.bartlett_window(window_length.item(), periodic=periodic.item())
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        N = window_length
        if N <= 1:
            return {"result": np.ones(N, dtype=np.float32)}

        if periodic:
            M = N + 1
        else:
            M = N

        n = tf.cast(tf.range(M), dtype=tf.float32)
        result = 2 * tf.divide(n, tf.cast(M - 1, dtype=tf.float32))
        result = 1 - tf.abs(result - 1)
        if periodic:
            result = result[:-1]

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "window_length": 10,
        "periodic": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "window_length": 5,
        "periodic": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()