import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.fftn(input_tensor, s=n, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    input_tensor = tf.constant(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", None)

    rank = len(input_tensor.shape)
    
    if dim < 0:
        dim = rank + dim

    if n is None:
        n = []
        for i in range(rank):
            n.append(input_tensor.shape[i])
        n = np.array(n)
    else:
        if isinstance(n, int):
            n = np.array([n])
        else:
            n = np.array(n)
    
    axes = [dim] if isinstance(dim, int) else [dim]

    fft_length = n[0] if len(n) > 0 else input_tensor.shape[dim]
    
    if axes[0] >= len(input_tensor.shape):
        pad_shape = list(input_tensor.shape)
        pad_shape[axes[0]] = fft_length
        padding = [(0,0)] * len(input_tensor.shape)
        padding[axes[0]] = (0,fft_length - input_tensor.shape[axes[0]])
        
        input_tensor = tf.pad(input_tensor, padding)
    else:
        input_shape = list(input_tensor.shape)
        if input_shape[axes[0]] > fft_length:
            slices = [slice(None)] * len(input_shape)
            slices[axes[0]] = slice(0, fft_length)
            input_tensor = input_tensor[tuple(slices)]

    result = tf.signal.fft(tf.cast(input_tensor, dtype=tf.complex64))
    
    if norm == "ortho":
        result = result / tf.cast(tf.math.sqrt(tf.cast(tf.shape(result)[dim], tf.float32)), tf.complex64)

    result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(8, 8).astype(np.float32),
        "n": 8,
        "dim": -1,
        "norm": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()