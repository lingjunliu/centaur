import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.hfft2(input_tensor, s=s, dim=dim, norm=norm)

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
        norm_str = input_dict.get("norm", "backward")

        rank = len(input_tensor.shape)
        axes = dim
        if axes is None:
            axes = list(range(rank))

        if isinstance(axes, int):
            axes = [axes]

        axes = [d if d >= 0 else d + rank for d in axes]
        axes = sorted(axes)

        if s is not None:
            output_shape = s
        else:
            output_shape = list(input_tensor.shape)
            output_shape[axes[-1]] = 2 * (input_tensor.shape[axes[-1]] - 1)
            output_shape = tuple(output_shape)

        result = tf.signal.rfft2d(input_tensor, fft_length=s)

        if norm_str == "forward":
            norm_factor = 1.0 / np.prod(output_shape)
            result = result * norm_factor
        elif norm_str == "ortho":
            norm_factor = 1.0 / np.sqrt(np.prod(output_shape))
            result = result * norm_factor

        if s is None:
            s = output_shape

        torch_output_shape = list(input_tensor.shape)
        torch_output_shape[axes[-1]] = s[axes[-1]] // 2 + 1
        torch_output_shape = tuple(torch_output_shape)

        result = result[..., :torch_output_shape[-1]]

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 9).astype(np.float32),
        "s": (10, 9),
        "dim": (-2, -1),
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()