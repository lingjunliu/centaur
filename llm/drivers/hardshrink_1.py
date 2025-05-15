import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    lambd = input_dict.get("lambd", 0.5)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.hardshrink(input_tensor, lambd=lambd)
    
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
        lambd = input_dict.get("lambd", 0.5)
        
        abs_input = tf.abs(input_tensor)
        mask = abs_input > lambd
        result = tf.where(mask, input_tensor, tf.zeros_like(input_tensor))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "lambd": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-2.0, -0.3, 0.1, 0.8, 1.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-2.0, -0.3, 0.1, 0.8, 1.5], dtype=np.float32),
        "lambd": 0.7
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()