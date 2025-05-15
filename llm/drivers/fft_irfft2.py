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

        rank = len(input_tensor.shape)
        axes = [d % rank for d in dim]
        axes = sorted(axes)

        if s is None:
            s_list = list(input_tensor.shape)
            for i in range(len(dim)):
                if dim[i] == -1 or dim[i] == rank - 1:
                     s_list[dim[i]] = 2 * (input_tensor.shape[dim[i]] - 1)
            s = tuple([s_list[i] for i in axes])
        
        result = tf.signal.irfft2d(input_tensor, fft_length=s)

        if norm == "forward":
            pass
        elif norm == "backward":
            n = 1.0
            for dim_val in s:
                n *= dim_val
            result = result / n
        elif norm == "ortho":
            n = 1.0
            for dim_val in s:
                n *= dim_val
            result = result / np.sqrt(n)
        else:
            raise ValueError(f"Invalid norm value: {norm}")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64),
        "s": (2, 3),
        "dim": (-2, -1),
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_res = torch_result["result"]
    tf_res = tf_result["result"]

    assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()