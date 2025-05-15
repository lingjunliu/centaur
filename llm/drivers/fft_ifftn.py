import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.ifftn(input_tensor, s=s, dim=dim, norm=norm)

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
        dim = input_dict.get("dim", None)
        norm = input_dict.get("norm", "backward")

        if s is None:
            if dim is None:
                s = input_tensor.shape
            else:
                s = [input_tensor.shape[d] for d in dim]

        if dim is None:
            dim = list(range(len(s)))

        temp = tf.cast(input_tensor, tf.complex128)

        result = temp
        for axis in sorted(dim):
            result = tf.signal.ifft(result)

        n = np.prod(s)

        if norm == "forward":
            scale = 1.0
        elif norm == "backward":
            scale = 1.0 / n
        elif norm == "ortho":
            scale = 1.0 / np.sqrt(n)
        else:
            raise ValueError(f"Invalid norm mode: {norm}")

        result = result * scale
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(8, 8).astype(np.complex64),
        "s": (8, 8),
        "dim": (0, 1),
        "norm": "backward"
    }

    input_data2 = {
        "input": np.random.rand(8, 8).astype(np.complex64),
        "s": (8, 8),
        "dim": (0, 1),
        "norm": "forward"
    }
    
    input_data3 = {
        "input": np.random.rand(8, 8).astype(np.complex64),
        "s": (8, 8),
        "dim": (0, 1),
        "norm": "ortho"
    }


    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    torch_result = torch_version(input_data2)
    tf_result = tensorflow_version(input_data2)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    torch_result = torch_version(input_data3)
    tf_result = tensorflow_version(input_data3)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()