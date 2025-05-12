import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.fft(input_tensor, n=n, dim=dim, norm=norm)
    
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
        n = input_dict.get("n", None)
        dim = input_dict.get("dim", -1)
        norm = input_dict.get("norm", "backward")

        input_tensor_rank = len(input_tensor.shape)
        dim = dim if dim >= 0 else input_tensor_rank + dim
        
        if n is not None:
            pad_shape = list(input_tensor.shape)
            pad_shape[dim] = max(0, n - input_tensor.shape[dim])
            if pad_shape[dim] > 0:
                padding = [[0, 0]] * input_tensor_rank
                padding[dim] = [0, pad_shape[dim]]
                input_tensor = tf.pad(input_tensor, padding, "CONSTANT")
            elif pad_shape[dim] < 0:
                slices = [slice(None)] * input_tensor_rank
                slices[dim] = slice(0, n)
                input_tensor = input_tensor[tuple(slices)]
        
        result = tf.signal.fft(tf.cast(input_tensor, tf.complex128))
        
        if norm == "forward":
            result = result / tf.cast(tf.shape(input_tensor)[dim], tf.complex128)
        elif norm == "ortho":
            result = result / tf.cast(tf.math.sqrt(tf.cast(tf.shape(input_tensor)[dim], tf.float64)), tf.complex128)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "n": 4,
        "dim": -1,
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.0 + 1.0j, 2.0 + 3.0j, 4.0 + 5.0j, 6.0 + 7.0j], dtype=np.complex64),
        "n": 4,
        "dim": -1,
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()