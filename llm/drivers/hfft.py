import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.hfft(input_tensor, n=n, dim=dim, norm=norm)
    
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

        input_tensor = tf.cast(input_tensor, tf.complex64)

        rank = len(input_tensor.shape)
        axes = dim if dim >= 0 else rank + dim
        
        if n is None:
            n = 2 * (input_tensor.shape[axes] - 1)
        
        hermitian_input = tf.concat([input_tensor, tf.reverse(tf.math.conj(input_tensor[..., 1:-1]), axis=[axes])], axis=-1)
        result = tf.signal.ifft(hermitian_input)
        
        if norm == "forward":
            result = result / tf.cast(n, tf.complex64)
        elif norm == "ortho":
            result = result / tf.math.sqrt(tf.cast(n, tf.complex64))

        result = tf.math.real(result).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.5 - 0j, -0.125 - 0.1720j, -0.125 - 0.0406j], dtype=np.complex64),
        "n": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    n = input_data.get("n", 2 * (len(input_data["input"]) - 1))
    
    torch_result_res = torch_result["result"]
    tf_result_res = tf_result["result"][:n]
    
    assert np.allclose(torch_result_res, tf_result_res, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5 - 0j, -0.125 - 0.1720j, -0.125 - 0.0406j], dtype=np.complex64),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    n = 2 * (len(input_data["input"]) - 1)
    
    torch_result_res = torch_result["result"]
    tf_result_res = tf_result["result"][:n]
    
    assert np.allclose(torch_result_res, tf_result_res, atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1+0j, 1+0j, 1+0j], dtype=np.complex64),
        "n": 5
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    n = input_data.get("n", 2 * (len(input_data["input"]) - 1))
    
    torch_result_res = torch_result["result"]
    tf_result_res = tf_result["result"][:n]
    
    assert np.allclose(torch_result_res, tf_result_res, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()