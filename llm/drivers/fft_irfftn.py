import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.irfftn(input_tensor, s=s, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.cast(input_tensor, dtype=tf.complex64)
        rank = len(input_tensor.shape)

        if dim is None:
            dim = list(range(rank))
        
        if s is None:
            s = []
            for i in range(len(dim)):
                s.append(2 * (input_tensor.shape[dim[i]] - 1))

        if len(s) < len(dim):
          raise ValueError("Shape parameter s must have at least as many dimensions as dim parameter.")

        if len(s) > len(dim):
          dim = list(range(rank))[-len(s):]
        
        axes = []
        for i in range(len(dim)):
          axes.append(dim[i])
        
        if s is not None:
            if len(s) > 0:
                fft_length = s[-1]
                result = tf.signal.irfft(input_tensor, fft_length=fft_length)
            else:
                result = tf.signal.irfft(input_tensor)
        else:
            result = tf.signal.irfft(input_tensor)


        n = np.prod(s) if s is not None else np.prod([2 * (input_tensor.shape[d] - 1) for d in dim])
        if norm == "forward":
            scale = 1.0
        elif norm == "backward":
            scale = 1.0 / n
        elif norm == "ortho":
            scale = 1.0 / np.sqrt(n)
        else:
            raise ValueError(f"Invalid normalization mode: {norm}")

        result = result * scale

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64),
        "s": (2, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64),
        "dim": (1,)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64),
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()