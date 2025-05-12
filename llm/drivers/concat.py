import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    tensors = [torch.tensor(t) for t in input_dict["tensors"]]
    dim = input_dict.get("dim", 0)
    
    if not cpu:
        tensors = [t.cuda() for t in tensors]
    
    result = torch.concat(tensors, dim=dim)
    
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
        tensors = [tf.constant(t) for t in input_dict["tensors"]]
        dim = input_dict.get("dim", 0)
        
        result = tf.concat(tensors, axis=dim)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "tensors": [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32)
        ],
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "tensors": [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32)
        ],
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()