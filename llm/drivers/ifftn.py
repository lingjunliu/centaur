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
    
    result = torch.fft.ifftn(input_tensor, s=s, dim=dim, norm=norm)
    
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
        input_tensor = tf.constant(input_dict["input"])
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", None)
        norm = input_dict.get("norm", "backward")

        input_rank = len(input_tensor.shape)

        if dim is None:
            if s is None:
                dim = list(range(input_rank))
            else:
                dim = list(range(input_rank - len(s), input_rank))
        
        if s is None:
            s = [input_tensor.shape[d] for d in dim]

        if len(s) != len(dim):
            raise ValueError("The lengths of s and dim must match")

        padded_input = input_tensor
        for i, d in enumerate(dim):
            target_len = s[i]
            current_len = input_tensor.shape[d]
            
            if target_len == -1:
                continue

            if target_len > current_len:
                pad_width = [(0, 0)] * input_rank
                pad_width[d] = (0, target_len - current_len)
                padded_input = tf.pad(padded_input, pad_width)
            elif target_len < current_len:
                slices = [slice(None)] * input_rank
                slices[d] = slice(0, target_len)
                padded_input = padded_input[tuple(slices)]
        
        result = tf.cast(padded_input, tf.complex128)
        
        for d in sorted(dim):
          result = tf.signal.ifft(result)

        n = 1.0
        for dim_idx, dim_size in enumerate(s):
            if dim_size != -1:
                n *= dim_size
            else:
                n *= input_tensor.shape[dim[dim_idx]]

        if norm == "forward":
            pass
        elif norm == "backward":
            result = result / n
        elif norm == "ortho":
            result = result / np.sqrt(n)
        else:
            raise ValueError(f"Invalid norm value: {norm}")
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(4, 4).astype(np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(4, 4, 4).astype(np.complex64),
        "s": (2, 2, 2),
        "dim": (0, 1, 2),
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(4, 4, 4).astype(np.complex64),
        "s": (8, 8),
        "dim": (1, 2),
        "norm": "forward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()