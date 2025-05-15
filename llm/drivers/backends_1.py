import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"])
    alpha = input_dict.get("alpha", 1.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    result = torch.add(input_tensor, other_tensor, alpha=alpha)
    
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
        other_tensor = tf.constant(input_dict["other"])
        alpha = input_dict.get("alpha", 1.0)
        
        result = tf.add(input_tensor, tf.multiply(other_tensor, alpha))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "other": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "alpha": 1
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()