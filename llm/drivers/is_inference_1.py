import numpy as np
import torch

_is_inference = False

def torch_version(input_dict, cpu=True):
    global _is_inference
    
    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    _is_inference = input_dict["mode"]
    
    result = _is_inference

    if not cpu:
        result = torch.tensor(result).cpu()
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    torch_mode = input_dict["mode"]
    
    tf_mode = torch_mode
    if torch_mode == True:
        tf_mode = True
    elif torch_mode == False:
        tf_mode = False
    else:
        tf_mode = bool(torch_mode)

    if cpu:
        with tf.device("/cpu:0"):
           result = tf_mode
    else:
        with tf.device("/gpu:0"):
           result = tf_mode

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "mode": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()