import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.ihfftn(input_tensor, s=s, dim=dim, norm=norm)

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

        input_tensor_complex = tf.complex(input_tensor, tf.zeros_like(input_tensor))

        if dim is None:
            if s is None:
                dim = list(range(len(input_tensor.shape)))
            else:
                dim = list(range(len(input_tensor.shape) - len(s), len(input_tensor.shape)))

        if s is None:
            s = [input_tensor.shape[d] for d in dim]

        axes = dim
        result = input_tensor_complex
        for axis in reversed(axes):
            N = tf.cast(tf.shape(result)[axis], tf.complex64)
            result = tf.signal.fft(result)

        result = tf.math.real(result)
        if input_tensor.shape[-1] % 2 == 0:
            result = result[..., :input_tensor.shape[-1]//2 + 1]
        else:
            result = result[..., :(input_tensor.shape[-1]+1)//2]

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
        "s": None,
        "dim": None,
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()