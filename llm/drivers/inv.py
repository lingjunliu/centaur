import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    
    if not cpu:
        A = A.cuda()
    
    try:
        Ainv = torch.linalg.inv(A)
    except RuntimeError:
        return {"result": None}
    
    if not cpu:
        Ainv = Ainv.cpu()
    
    return {"result": Ainv.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        
        try:
            Ainv = tf.linalg.inv(A)
        except tf.errors.InvalidArgumentError:
            return {"result": None}
        
        result = Ainv.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "A": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    if torch_result["result"] is not None and tf_result["result"] is not None:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    if torch_result["result"] is not None and tf_result["result"] is not None:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    if torch_result["result"] is None and tf_result["result"] is None:
        pass
    else:
        assert False, "Invertible check failed"

    print("Success")

if __name__ == "__main__":
    main()