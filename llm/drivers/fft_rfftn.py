import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.rfftn(input_tensor, s=s, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if dim is None:
        dim = list(range(len(input_tensor.shape)))
    elif not isinstance(dim, list):
        dim = [dim]

    if s is None:
        s = [input_tensor.shape[i] for i in dim]

    input_tensor_tf = input_tensor

    def get_transpose_pattern(axis, rank):
        pattern = list(range(rank))
        pattern.pop(axis)
        pattern.append(axis)
        return pattern

    if len(dim) > 0:
        axes = sorted(dim)

        for axis_index in range(len(axes)):
            axis = axes[axis_index]
            target_length = s[axis_index]

            if target_length != -1 and target_length != input_tensor_tf.shape[axis]:
                pad_size = target_length - input_tensor_tf.shape[axis]

                if pad_size > 0:
                    paddings = [[0, 0]] * len(input_tensor_tf.shape)
                    paddings[axis] = [0, pad_size]
                    input_tensor_tf = tf.pad(input_tensor_tf, paddings, "CONSTANT")
                else:
                    slices = [slice(None)] * len(input_tensor_tf.shape)
                    slices[axis] = slice(0, target_length)
                    input_tensor_tf = input_tensor_tf[tuple(slices)]

        fft_result = input_tensor_tf
        for axis in reversed(axes):
            fft_length = tf.constant(input_tensor_tf.shape[axis], dtype=tf.int32)
            fft_result = tf.signal.rfft(fft_result, fft_length=[fft_length])
            fft_result = tf.transpose(fft_result, perm=get_transpose_pattern(axis, len(fft_result.shape)))

        if norm == "forward":
            n = 1.0
            for length in s:
                if length != -1:
                  n *= length
            fft_result = fft_result / n
        elif norm == "ortho":
            n = 1.0
            for length in s:
                if length != -1:
                  n *= length
            fft_result = fft_result / np.sqrt(n)

        result = fft_result.numpy()
    else:
        result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
        "s": (10,10),
        "dim": [0,1],
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
        "s": (10,10),
        "dim": [0,1],
        "norm": "forward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
        "s": (10,10),
        "dim": [0,1],
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()