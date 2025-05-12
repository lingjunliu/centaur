import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.rfft(input_tensor, n=n, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if n is not None:
            input_tensor = tf.pad(input_tensor, [[0, max(0, n - input_tensor.shape[0])]])
            input_tensor = input_tensor[:n]
        
        input_tensor = tf.signal.fft(tf.cast(input_tensor, dtype=tf.complex64))
        input_tensor = input_tensor[:input_tensor.shape[0]//2 + 1]

        if norm == "forward":
            input_tensor = input_tensor / tf.cast(n if n is not None else input_tensor.shape[0]*1.0 , tf.complex64)
        elif norm == "ortho":
            input_tensor = input_tensor / tf.cast(tf.sqrt(tf.cast(n if n is not None else input_tensor.shape[0]*1.0, tf.float32)), tf.complex64)
            
        result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0, 1, 2, 3], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "n": 4,
        "norm": "forward"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "n": 8,
        "norm": "ortho"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    

    print("Success")

if __name__ == "__main__":
    main()