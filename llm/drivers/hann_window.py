import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    window_length = int(input_dict["window_length"])
    periodic = input_dict.get("periodic", True)

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    result = torch.hann_window(window_length, periodic=periodic, device=device)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    window_length = int(input_dict["window_length"])
    periodic = input_dict.get("periodic", True)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if window_length == 1:
          result = tf.ones(1)
        else:
          if periodic:
            multiplier = tf.range(window_length, dtype=tf.float32) / (window_length)
          else:
            multiplier = tf.range(window_length, dtype=tf.float32) / (window_length - 1)
          
          result = 0.5 - 0.5 * tf.cos(multiplier * 2 * np.pi)
        
        result = result.numpy()
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
    
    input_data = {
        "window_length": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()