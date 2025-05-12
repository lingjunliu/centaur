import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)

    result = torch.bartlett_window(window_length, periodic=periodic)
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    window_length = input_dict["window_length"]
    periodic = input_dict.get("periodic", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        window_length = tf.constant(window_length, dtype=tf.int32)
        
        if periodic:
            N = window_length
        else:
            N = window_length + 1
        
        n = tf.range(N)
        n = tf.cast(n, dtype=tf.float32)
        N = tf.cast(N, dtype=tf.float32)
        result = 2 / (N - 1) * ( (N - 1) / 2 - tf.abs(n - (N - 1) / 2) )

        if not periodic:
            result = result[:-1]
        
        result = tf.cast(result, dtype=tf.float64)

        result = result.numpy()
        result = result.astype(np.float32)

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "window_length": 5,
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