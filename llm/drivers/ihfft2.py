import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.ihfft2(input_tensor, s=s, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", (-2, -1))
        norm = input_dict.get("norm", "backward")

        input_shape = input_tensor.shape
        axes = [d % len(input_shape) for d in dim]
        axes = sorted(axes)

        if s is None:
            s = [input_shape[d] for d in axes]

        if len(axes) != 2:
            raise ValueError("tensorflow_version only supports 2 dimensions")
        
        if norm == "forward":
            scale = 1.0
        elif norm == "backward":
            scale = 1.0 / np.prod(s)
        elif norm == "ortho":
            scale = 1.0 / np.sqrt(np.prod(s))
        else:
            raise ValueError("Invalid norm value")

        fft_result = tf.signal.ifft2d(tf.complex(real=input_tensor, imag=tf.zeros_like(input_tensor)))
        result = tf.cast(tf.math.real(fft_result), dtype=tf.float32) * scale

        if input_tensor.shape[-1] % 2 != 0:
            result = result
        else:
          target_length = input_tensor.shape[-1] // 2 + 1
          result = result[..., :target_length]


        return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 6).astype(np.float32),
        "s": (10, 12),
        "dim": (-2, -1),
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()