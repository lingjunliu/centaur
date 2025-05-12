import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.rfft2(input_tensor, s=s, dim=dim, norm=norm)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", (-2, -1))
        norm = input_dict.get("norm", "backward")

        rank = len(input_tensor.shape)
        axes = [d % rank for d in dim]

        if s is not None:
            pad_shape = list(input_tensor.shape)
            for i, axis in enumerate(axes):
                if s[i] > input_tensor.shape[axis]:
                    pad_size = s[i] - input_tensor.shape[axis]
                    padding = [[0, 0]] * rank
                    padding[axis] = [0, pad_size]
                    input_tensor = tf.pad(input_tensor, padding)
                elif s[i] < input_tensor.shape[axis]:
                    slices = [slice(None)] * rank
                    slices[axis] = slice(0, s[i])
                    input_tensor = input_tensor[tuple(slices)]

        fft_result = tf.signal.fft2d(tf.complex(input_tensor, tf.zeros_like(input_tensor)))

        nyquist_indices = []
        for axis in axes:
            if axis == axes[-1]:
                nyquist_indices.append(tf.shape(input_tensor)[axis] // 2 + 1)
            else:
                nyquist_indices.append(tf.shape(input_tensor)[axis])
        
        slices = [slice(None)] * len(fft_result.shape)
        for i, axis in enumerate(axes):
            slices[axis] = slice(0, nyquist_indices[i].numpy())

        fft_result = fft_result[tuple(slices)]

        if norm == "forward":
            n = 1.0
            if s is not None:
                n = np.prod(s)
            else:
                n = np.prod([tf.shape(input_tensor)[axis].numpy() for axis in axes])
            fft_result = fft_result / n
        elif norm == "ortho":
            n = 1.0
            if s is not None:
                n = np.prod(s)
            else:
                n = np.prod([tf.shape(input_tensor)[axis].numpy() for axis in axes])
            fft_result = fft_result / np.sqrt(n)

        result = tf.math.real(fft_result).numpy()

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
        "input": np.random.rand(5, 7).astype(np.float32),
        "s": (8, 10),
        "dim": (-2, -1),
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()