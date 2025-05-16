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

    input_tensor = tf.constant(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        rank = len(input_tensor.shape)

        if isinstance(dim, list):
            axes = dim
        else:
            if dim < 0:
                dim = rank + dim
            axes = [dim] if isinstance(dim, int) else [dim]

        axes = [i % rank for i in axes]

        if n is None:
            n = [tf.shape(input_tensor)[i] for i in axes]
        elif isinstance(n, int):
            n = [n]
        
        if len(axes) != len(n):
            raise ValueError("Shape of n and axes must match.")

        
        result = tf.signal.fft2d(tf.cast(input_tensor, tf.complex64))
        
        if norm == "ortho":
            size = tf.cast(tf.reduce_prod([tf.cast(x, tf.float32) for x in n]), tf.float32)
            result = result / tf.sqrt(size)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1+1j, -1-1j], [-1+1j, 1-1j]], dtype=np.complex64),
        "n": [2, 2],
        "dim": [0, 1],
        "norm": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()