import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.ihfft(input_tensor, n=n, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.resolve_conj().numpy()}

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

        input_shape = input_tensor.shape
        dim = dim if dim >= 0 else len(input_shape) + dim

        if n is None:
            n = input_shape[dim] * 2 - 2 if input_shape[dim] > 1 else 1
        
        input_tensor = tf.cast(input_tensor, tf.complex64)

        fft_result = tf.signal.irfft(input_tensor, fft_length=[n])

        if norm == "forward":
            pass
        elif norm == "backward":
            if n is not None:
                fft_result = fft_result / tf.cast(tf.complex(tf.cast(n, input_tensor.dtype.real_dtype), 0.0), input_tensor.dtype)
        elif norm == "ortho":
            if n is not None:
                fft_result = fft_result / tf.cast(tf.complex(tf.sqrt(tf.cast(n, input_tensor.dtype.real_dtype)), 0.0), input_tensor.dtype)

        result = fft_result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0, 1, 2, 3, 4], dtype=np.float32),
        "n": 5,
        "dim": -1,
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()