import numpy as np
import tensorflow as tf
import inspect
import torch

def torch_version(input_dict, cpu=True):
    window_length = input_dict["window_length"]
    beta = input_dict.get("beta", 12.0)
    sym = input_dict.get("sym", True)
    
    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    result = torch.kaiser_window(window_length, periodic=not sym, beta=beta)
    if not cpu:
        result = result.to(device)
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    window_length = input_dict["window_length"]
    beta = input_dict.get("beta", 12.0)
    sym = input_dict.get("sym", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if 'symmetric' in inspect.getfullargspec(tf.signal.kaiser_window).args:
            result = tf.signal.kaiser_window(window_length, beta=beta, symmetric=sym)
        else:
            result = tf.signal.kaiser_window(window_length, beta=beta)
        
        if not sym:
          n = tf.cast(window_length, dtype=tf.float32)
          window = (tf.range(n) + 0.5) * (2 / n) - 1
        else:
          n = tf.cast(window_length, dtype=tf.float32)
          window = tf.range(n) * (2 / (n - 1)) - 1

        result = tf.signal.kaiser_window(window_length, beta=beta)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "window_length": 5,
        "beta": 5.0,
        "sym": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "window_length": 6,
        "beta": 5.0,
        "sym": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()