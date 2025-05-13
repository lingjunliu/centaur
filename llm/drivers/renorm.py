import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict["p"]
    dim = input_dict["dim"]
    maxnorm = input_dict["maxnorm"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.renorm(input_tensor, p, dim, maxnorm)
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict["p"]
        dim = input_dict["dim"]
        maxnorm = input_dict["maxnorm"]
        
        norm = tf.norm(input_tensor, ord=p, axis=dim, keepdims=True)
        
        factor = tf.math.minimum(maxnorm / norm, 1.0)
        
        result = input_tensor * factor
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 1.0, 1.0], [6.0, 6.0, 6.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        "p": 2,
        "dim": 0,
        "maxnorm": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()