import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.irfft2(input_tensor, s=s, dim=dim, norm=norm)

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

        input_tensor_rank = len(input_tensor.shape)
        dim = [d if d >= 0 else input_tensor_rank + d for d in dim]

        axes = sorted(dim)

        if len(axes) != 2:
            raise ValueError("Only 2 dimensions are supported for irfft2 in tensorflow implementation.")
        
        fft_length = s

        result = tf.signal.irfft2d(input_tensor, fft_length=fft_length)

        if norm == "forward":
            norm_factor = 1.0
        elif norm == "backward":
            if s is None:
                norm_shape = list(result.shape)
                n = np.prod(norm_shape).astype(np.float32)
            else:
                n = np.prod(s).astype(np.float32)
            norm_factor = 1.0 / n
        elif norm == "ortho":
            if s is None:
                norm_shape = list(result.shape)
                n = np.sqrt(np.prod(norm_shape)).astype(np.float32)
            else:
                n = np.sqrt(np.prod(s)).astype(np.float32)
            norm_factor = 1.0 / n
        else:
            raise ValueError(f"Unsupported normalization mode: {norm}")
        
        result = result * norm_factor
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1+1j, 1-1j], [1+1j, 1-1j]], dtype=np.complex64),
        "s": (2, 2),
        "dim": (-2, -1),
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1+1j, 1-1j, 2], [1+1j, 1-1j, 2]], dtype=np.complex64),
        "s": (2, 4),
        "dim": (-2, -1),
        "norm": "backward"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()