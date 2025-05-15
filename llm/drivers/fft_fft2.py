import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.fft2(input_tensor, s=s, dim=dim, norm=norm)

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
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", (-2, -1))
        norm = input_dict.get("norm", "backward")
        
        input_shape = input_tensor.shape
        
        if s is None:
            s = [input_shape[d] for d in dim]

        axes = [d % len(input_shape) for d in dim] 
        
        if norm == "forward":
            normalization = "by_fft_length"
        elif norm == "backward":
            normalization = None
        elif norm == "ortho":
            normalization = "by_unit_norm"
        else:
            raise ValueError(f"Invalid norm: {norm}")
            
        result = tf.signal.fft2d(tf.cast(input_tensor, tf.complex64))
        
        if normalization == "by_fft_length":
            n = 1.0
            for length in s:
                n *= length
            result = result / n
        elif normalization == "by_unit_norm":
            n = 1.0
            for length in s:
                n *= length
            result = result / np.sqrt(n)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 5).astype(np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()